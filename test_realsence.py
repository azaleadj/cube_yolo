import pyrealsense2 as rs
import numpy as np
import cv2

pipline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 
profile = pipline.start(config) 
print("RealSense camera is connected.")

try:
    while True:
        frames = pipline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # images = np.hstack((color_image, depth_colormap))

        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('depth', depth_image)
        cv2.imshow('color', color_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
finally:
    pipline.stop()
    cv2.destroyAllWindows() 

