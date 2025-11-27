#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import yaml
import os
import time
import csv
from ultralytics import YOLO
from collections import deque

# 
# configurations
#
SAVE_DIR = "/home/acis/cube_yolo"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = f"{SAVE_DIR}/best.pt"
PLANE_FILE = f"{SAVE_DIR}/table_plane.yaml"

CUBE_KEYWORD = "Cube"
CONF_THRESHOLD = 0.75
STABLE_FRAMES = 5
XYZ_BUFFER_SIZE = 10
PLANE_THRESHOLD = 0.02   # 2 cm 阈值

window_name = "YOLO + RealSense"
model = YOLO(MODEL_PATH)

#
#  read write plane
#
def save_plane(A, B, C, D):
    with open(PLANE_FILE, "w") as f:
        yaml.dump({"A": float(A), "B": float(B), "C": float(C), "D": float(D)}, f)
    print(f"[INFO] Saved plane: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f}=0")


def load_plane():
    if not os.path.exists(PLANE_FILE):
        return None
    with open(PLANE_FILE, "r") as f:
        p = yaml.safe_load(f)
    print(f"[INFO] Loaded plane: {p['A']:.4f}x + {p['B']:.4f}y + {p['C']:.4f}z + {p['D']:.4f}=0")
    return np.array([p["A"], p["B"], p["C"], p["D"]])


plane = load_plane()

#
# RealSense initialize
# 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# 
#  click points on desk top
# 
clicked_points = []  # pixel points

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"[INFO] Clicked ({x}, {y}) — total: {len(clicked_points)}")


def fit_plane(points_xyz):
    pts = np.array(points_xyz)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    _, _, vh = np.linalg.svd(pts_centered)
    normal = vh[-1]
    A, B, C = normal
    D = -normal.dot(centroid)
    return A, B, C, D


# 
# Video & CSV
# 
ts = time.strftime("%Y%m%d-%H%M%S")
video_path = f"{SAVE_DIR}/yolo_bottompoint_{ts}.mp4"
csv_path   = f"{SAVE_DIR}/yolo_bottompoint_{ts}.csv"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["t","conf","u","v","depth","X","Y","Z","dist","on_table","finalX","finalY","finalZ"])

print(f"[INFO] Saving video → {video_path}")
print(f"[INFO] Saving CSV   → {csv_path}")

# 
# 稳定检测缓存 stable detection buffers
# 
history = deque(maxlen=STABLE_FRAMES)
xyz_buffer = deque(maxlen=XYZ_BUFFER_SIZE)

#
# GUI setup
# 
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, mouse_callback)

print("[INFO] Press 'p' to calibrate plane (click 10 desk points)")
print("[INFO] Press 'q' to quit")


# ==================================================
# Main loop
# ==================================================
try:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        color_img = np.asanyarray(color_frame.get_data())

        # ========== Plane calibration ==========
        if key == ord('p'):
            if len(clicked_points) < 5:
                print("[WARN] Need at least 5 table points!")
            else:
                xyz_pts = []
                for (u, v) in clicked_points:
                    z = depth_frame.get_distance(u, v)
                    if z > 0:
                        X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], z)
                        xyz_pts.append([X, Y, Z])

                if len(xyz_pts) >= 5:
                    A, B, C, D = fit_plane(xyz_pts)
                    plane = np.array([A, B, C, D])
                    save_plane(A, B, C, D)
                    print("[INFO] Plane calibrated.")
                else:
                    print("[WARN] Too few valid points.")

            clicked_points.clear()

        # If the plane is not yet calibrated, prompt the user
        if plane is None:
            cv2.putText(color_img, "Press 'p' and click table points",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(window_name, color_img)
            continue

        # ========== YOLO detection ==========
        results = model(color_img, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        annotated = results[0].plot()

        cube_info = None
        A, B, C, D = plane

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            if CUBE_KEYWORD.lower() in label.lower():
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                #  Use bottom edge midpoint (core modification) 
                u = int((x1 + x2) / 2)
                v = int(y2)    # bottom edge

                z = depth_frame.get_distance(u, v)
                if z <= 0:
                    continue

                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], z)

                # Distance to plane
                dist = abs(A*X + B*Y + C*Z + D) / np.sqrt(A*A + B*B + C*C)
                on_table = (dist < PLANE_THRESHOLD)

                cube_info = (conf, u, v, z, X, Y, Z, dist, on_table)
                break

        # Coordinate determination logic 
        if cube_info:
            conf,u,v,z,X,Y,Z,dist,on_table = cube_info

            print(f"[Cube] {'ON TABLE' if on_table else 'NOT on table'} "
                  f"| conf={conf:.2f} dist={dist:.3f}m XYZ=({X:.3f},{Y:.3f},{Z:.3f})")

            csv_writer.writerow([
                time.time(), conf, u, v, z, X, Y, Z,
                dist, 1 if on_table else 0, "", "", ""
            ])

            out_video.write(annotated)

            if on_table:
                history.append(True)
                xyz_buffer.append(np.array([X, Y, Z]))
            else:
                history.append(False)

            # Stable detection
            if len(history) == STABLE_FRAMES and all(history):
                finalXYZ = np.mean(list(xyz_buffer)[-5:], axis=0)
                fX, fY, fZ = finalXYZ
                print(f"\nFINAL POSITION LOCKED → X={fX:.3f}, Y={fY:.3f}, Z={fZ:.3f}\n")

                csv_writer.writerow([
                    time.time(), conf, u, v, z, X, Y, Z,
                    dist, 1, fX, fY, fZ
                ])
                break

        cv2.imshow(window_name, annotated)

finally:
    pipeline.stop()
    out_video.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")
