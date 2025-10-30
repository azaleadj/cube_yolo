

from ultralytics import  YOLO
import pyrealsense2 as rs
import numpy as np
import cv2  
from collections import deque
import time

model = YOLO('/home/acis/cube_yolo/best.pt')

pipline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 
profile = pipline.start(config)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = time.strftime("%Y%m%d-%H%M%S")
out = cv2.VideoWriter(f'/home/acis/cube_yolo/yolo_realsence_output_{timestamp}.mp4', fourcc, 20.0, (640, 480))
if not out.isOpened():
    print("Error: Could not open video for writing.")
else:
    print("Video opened successfully.")
output_filename = f'yolo_realsence_output_{timestamp}.mp4'

CONF_THRESHOLD = 0.7
Stable_frames = 10
recent_detections = deque(maxlen=Stable_frames)     
stable_detected = False

print("YOLO + Realsence started.Press 'q' to exit")

try:
    while True:
        frames = pipline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image, imgsz=640, conf=0.5, verbose=False) 

        annotated_frame = results[0].plot()

        detected = False
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            if ("Cube" in label) and conf >= CONF_THRESHOLD:
                detected = True
                break   
        recent_detections.append(detected)
        if len(recent_detections) == Stable_frames:
            if all(recent_detections) and not stable_detected:
                stable_detected = True
                print("Stable detection of Cube confirmed.")
            elif not any(recent_detections):
                stable_detected = False
        out.write(annotated_frame)
        cv2.imshow("YOLO + Realsence", annotated_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
finally:
    out.release()
    pipline.stop()
    cv2.destroyAllWindows()
    print("Video saved and YOLO + Realsence stopped.")
