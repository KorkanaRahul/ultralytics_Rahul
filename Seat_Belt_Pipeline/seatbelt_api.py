#!/usr/bin/env python3
"""
Seatbelt pipeline:
1) Detect vehicles in frame (vehicle_model)
2) For each vehicle crop -> detect windshield (windshield_model)
3) Crop windshield -> split left/right -> detect seatbelt (seatbelt_model)
4) Visualize results on original frame
Supports: image file, video file, realtime stream (webcam or RTSP)
"""
# Monkey-patch: make conv.WeightedAdd = block.WeightedAdd
import ultralytics.nn.modules.conv as conv_mod
import ultralytics.nn.modules.block as block_mod

conv_mod.WeightedAdd = block_mod.WeightedAdd

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Tuple
import tkinter as tk
import os
import time

# -------------------------
# Utility functions
# -------------------------
def save_annotated_image(image, input_path=None, output_dir="./outputs/images"):
    """
    Saves the annotated image to disk.

    Args:
        image (np.ndarray): Annotated OpenCV image
        input_path (str, optional): Original image path (used for naming)
        output_dir (str): Directory to save output
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    if input_path:
        base = os.path.splitext(os.path.basename(input_path))[0]
    else:
        base = f"frame_{int(time.time() * 1000)}"

    output_path = os.path.join(output_dir, f"{base}_annotated_ECA.jpg")

    cv2.imwrite(output_path, image)
    print(f"[INFO] Annotated image saved at: {output_path}")

    return output_path

def get_video_writer(cap, output_dir="./outputs/videos", name_prefix="output"):
    os.makedirs(output_dir, exist_ok=True)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out_path = os.path.join(
        output_dir, f"{name_prefix}.mp4"
    )

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    return writer, out_path

def draw_boxes(img, boxes: List[Tuple[int,int,int,int]], color=(0,255,0), label=""):
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(img, label, (x1, max(y1-6,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def boxes_from_result(results, conf_threshold=0.25):
    """
    Returns list of boxes [(x1,y1,x2,y2,conf), ...] above threshold
    """
    if results is None or len(results) == 0:
        return []

    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return []

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    out = []
    for box, conf in zip(boxes_xyxy, confs):
        if conf >= conf_threshold:
            x1, y1, x2, y2 = box
            out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return out


def best_box_from_result(results, conf_threshold=0.25):
    """
    Returns:
        boxes: [(x1,y1,x2,y2)] OR []
        conf : float OR None
    """
    if results is None or len(results) == 0:
        return [], None

    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return [], None

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    if best_conf < conf_threshold:
        return [], None

    x1, y1, x2, y2 = boxes_xyxy[best_idx]
    return [(int(x1), int(y1), int(x2), int(y2))], best_conf


# -------------------------
# Core processing
# -------------------------
class SeatbeltPipeline:
    def __init__(self,
                 vehicle_weights: str,
                 windshield_weights: str,
                 seatbelt_weights: str,
                 conf_vehicle=0.5,
                 conf_windshield=0.5,
                 conf_seatbelt=0.25):

        self.vehicle_model = YOLO(vehicle_weights)
        self.windshield_model = YOLO(windshield_weights)
        self.seatbelt_model = YOLO(seatbelt_weights)

        self.conf_vehicle = conf_vehicle
        self.conf_windshield = conf_windshield
        self.conf_seatbelt = conf_seatbelt

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        orig = frame.copy()
        h, w = orig.shape[:2]

        # 1) Vehicle detection
        veh_results = self.vehicle_model.predict(frame, conf=self.conf_vehicle, verbose=False)
        vehicle_boxes = boxes_from_result(veh_results, self.conf_vehicle)

        if not vehicle_boxes:
            cv2.putText(orig, "No vehicle detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return orig

        for (vx1, vy1, vx2, vy2, v_conf) in vehicle_boxes:
            vx1, vy1 = max(0,vx1), max(0,vy1)
            vx2, vy2 = min(w-1,vx2), min(h-1,vy2)

            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue

            cv2.rectangle(orig, (vx1,vy1), (vx2,vy2), (255,255,0), 2)
            cv2.putText(orig, f"Vehicle {v_conf:.2f}", (vx1, max(vy1-6,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # 2) Windshield detection
            wind_results = self.windshield_model.predict(vehicle_crop,
                                                         conf=self.conf_windshield,
                                                         verbose=False)

            wind_boxes, w_conf = best_box_from_result(wind_results, self.conf_windshield)

            if not wind_boxes:
                cv2.putText(orig, "No windshield", (vx1, vy2+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                continue

            wx1, wy1, wx2, wy2 = wind_boxes[0]
            awx1, awy1 = vx1 + wx1, vy1 + wy1
            awx2, awy2 = vx1 + wx2, vy1 + wy2

            cv2.rectangle(orig, (awx1,awy1), (awx2,awy2), (0,165,255), 2)
            cv2.putText(orig, f"Windshield {w_conf:.2f}",
                        (awx1, max(awy1-6,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

            windshield_crop = vehicle_crop[wy1:wy2, wx1:wx2]
            if windshield_crop.size == 0:
                continue

            # 3) Split windshield
            wh, ww = windshield_crop.shape[:2]
            mid = ww // 2
            left = windshield_crop[:, :mid]
            right = windshield_crop[:, mid:]

            # 4) Seatbelt detection
            left_res = self.seatbelt_model.predict(left,
                                                   conf=self.conf_seatbelt,
                                                   verbose=False)
            right_res = self.seatbelt_model.predict(right,
                                                    conf=self.conf_seatbelt,
                                                    verbose=False)

            left_boxes, sl_conf = best_box_from_result(left_res, self.conf_seatbelt)
            right_boxes, sr_conf = best_box_from_result(right_res, self.conf_seatbelt)

            if not left_boxes:
                cv2.putText(orig, "No seatbelt (Driver)",
                            (awx1, awy2+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            if not right_boxes:
                cv2.putText(orig, "No seatbelt (Passenger)",
                            (awx1 + mid, awy2+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            for (lx1, ly1, lx2, ly2) in left_boxes:
                ox1 = vx1 + wx1 + lx1
                oy1 = vy1 + wy1 + ly1
                ox2 = vx1 + wx1 + lx2
                oy2 = vy1 + wy1 + ly2
                cv2.rectangle(orig, (ox1,oy1), (ox2,oy2), (0,255,0), 2)
                cv2.putText(orig, f"Seatbelt (Driver) {sl_conf:.2f}",
                            (ox1, max(oy1-6,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            for (rx1, ry1, rx2, ry2) in right_boxes:
                ox1 = vx1 + wx1 + mid + rx1
                oy1 = vy1 + wy1 + ry1
                ox2 = vx1 + wx1 + mid + rx2
                oy2 = vy1 + wy1 + ry2
                cv2.rectangle(orig, (ox1,oy1), (ox2,oy2), (0,255,0), 2)
                cv2.putText(orig, f"Seatbelt (Passenger) {sr_conf:.2f}",
                            (ox1, max(oy1-6,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return orig


# -------------------------
# I/O functions
# -------------------------
def run_on_image(pipeline, source_path, out_path=None):
    frame = cv2.imread(source_path)
    if frame is None:
        raise FileNotFoundError(source_path)

    annotated = pipeline.process_frame(frame)

    # Save annotated image
    if out_path:
        cv2.imwrite(out_path, annotated)
    else:
        save_annotated_image(annotated, input_path=source_path)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



# def run_on_video(pipeline, source_path, out_path=None):
#     cap = cv2.VideoCapture(source_path)
#     writer = None

#     if out_path:
#         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS) or 25
#         writer = cv2.VideoWriter(out_path,
#                                  cv2.VideoWriter_fourcc(*'mp4v'),
#                                  fps, (w,h))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         annotated = pipeline.process_frame(frame)
#         if writer:
#             writer.write(annotated)

#         cv2.imshow("Video", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     if writer:
#         writer.release()
#     cv2.destroyAllWindows()

def run_on_video(pipeline, source_path, out_path=None):
    cap = cv2.VideoCapture(source_path)
    name = os.path.splitext(os.path.basename(source_path))[0]
    print(f"Processing video: {source_path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    if out_path:
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            cap.get(cv2.CAP_PROP_FPS) or 25,
            (int(cap.get(3)), int(cap.get(4)))
        )
    else:
        writer, out_path = get_video_writer(cap, name_prefix=f"annotated_{name}")

    while cap.isOpened():
        ret, frame = cap.read()
        print("Progress: "f"{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{num_frames}", end='\r')
        if not ret:
            break

        annotated = pipeline.process_frame(frame)
        writer.write(annotated)

        # cv2.imshow("Video", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    print("Video processing complete.")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()



# def run_on_stream(pipeline, source=0):
#     cap = cv2.VideoCapture(source)
#     prev = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         annotated = pipeline.process_frame(frame)
#         fps = 1.0 / (time.time() - prev + 1e-6)
#         prev = time.time()

#         cv2.putText(annotated, f"FPS:{int(fps)}", (10,30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#         cv2.imshow("Stream", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

def run_on_stream(pipeline, source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open stream")

    writer, _ = get_video_writer(cap, name_prefix="stream")

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = pipeline.process_frame(frame)
        writer.write(annotated)

        fps = 1.0 / (time.time() - prev + 1e-6)
        prev = time.time()

        cv2.putText(annotated, f"FPS:{int(fps)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Stream", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()



# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image","video","stream"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--vehicle", default="./models/Vehicle_model.pt")
    parser.add_argument("--windshield", default="./models/windsheildWbest.pt")
    parser.add_argument("--seatbelt", default="./models/Seat_Belt_weights_ECA.pt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--conf_vehicle", type=float, default=0.5)
    parser.add_argument("--conf_windshield", type=float, default=0.5)
    parser.add_argument("--conf_seatbelt", type=float, default=0.25)
    args = parser.parse_args()

    pipeline = SeatbeltPipeline(
        args.vehicle,
        args.windshield,
        args.seatbelt,
        args.conf_vehicle,
        args.conf_windshield,
        args.conf_seatbelt
    )

    if args.mode == "image":
        run_on_image(pipeline, args.source, args.out)
    elif args.mode == "video":
        run_on_video(pipeline, args.source, args.out)
    else:
        src = 0 if args.source == "0" else args.source
        run_on_stream(pipeline, src)


if __name__ == "__main__":
    main()



# ## Example command to run on an image:
# python seatbelt_pipeline.py --mode image --source /content/drive/MyDrive/SeatBelt_project/DATA/ALL_Data/Car__018.jpg --vehicle /content/drive/MyDrive/SeatBelt_project/Weights/vehicle.pt --windshield /content/drive/MyDrive/SeatBelt_project/Weights/windsheildWbest.pt --seatbelt /content/drive/MyDrive/SeatBelt_project/Weights/Seat_Belt_weights_dwconv.pt
  


# ## Example command to run on a video:
# python seatbelt_pipeline.py --mode video --source /content/drive/MyDrive/input_video.mp4 --out /content/drive/MyDrive/output_video.mp4

# ## Example command to run on a webcam stream:
# python seatbelt_pipeline.py \
#   --mode stream \
#   --source 0

# ## Example command to run on an RTSP stream:
# python seatbelt_pipeline.py \
#   --mode stream \
#   --source rtsp://username:password@ip:port/stream


# import cv2
# import numpy as np
# from ultralytics import YOLO
# import torch

# # Check for GPU
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# class SeatbeltPipeline:
#     def __init__(self, vehicle_weights, windshield_weights, seatbelt_weights):
#         # Load Models
#         print("Loading models...")
#         self.model_vehicle = YOLO(vehicle_weights)    # Generic model (e.g., yolov8n.pt) for cars
#         self.model_windshield = YOLO(windshield_weights) # Custom model for windshield
#         self.model_seatbelt = YOLO(seatbelt_weights)     # Custom model for seatbelt
        
#         # Target classes for vehicle detection (based on COCO dataset standard)
#         # 2: car, 3: motorcycle, 5: bus, 7: truck
#         self.vehicle_classes = [2, 5, 7] 

#     def split_image(self, image):
#         """Splits the image into left and right halves."""
#         h, w, _ = image.shape
#         mid = w // 2
#         left_img = image[:, :mid, :]
#         right_img = image[:, mid:, :]
#         return left_img, right_img, mid

#     def draw_box(self, img, box, label, color, offset=(0, 0)):
#         """Draws a bounding box on the main image with coordinate offsets."""
#         x1, y1, x2, y2 = map(int, box)
#         off_x, off_y = offset
        
#         # Shift coordinates back to original frame
#         x1 += off_x
#         y1 += off_y
#         x2 += off_x
#         y2 += off_y

#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         return img

#     def process_frame(self, frame):
#         """
#         Main logic: Vehicle -> Windshield -> Split -> Seatbelt
#         """
#         frame_display = frame.copy()
        
#         # --- LAYER 1: Vehicle Detection ---
#         vehicle_results = self.model_vehicle.predict(frame, verbose=False, classes=self.vehicle_classes)
        
#         for v_box in vehicle_results[0].boxes.data.tolist():
#             vx1, vy1, vx2, vy2, conf, cls = map(float, v_box)
#             vx1, vy1, vx2, vy2 = map(int, [vx1, vy1, vx2, vy2])
            
#             # Draw Vehicle Box (Blue)
#             cv2.rectangle(frame_display, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
#             cv2.putText(frame_display, "Vehicle", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#             # Crop Vehicle
#             vehicle_crop = frame[vy1:vy2, vx1:vx2]
#             if vehicle_crop.size == 0: continue

#             # --- LAYER 2: Windshield Detection (on Vehicle Crop) ---
#             ws_results = self.model_windshield.predict(vehicle_crop, verbose=False)
            
#             if not ws_results[0].boxes:
#                 continue # No windshield found on this vehicle

#             # Assuming the largest box is the windshield if multiple detected
#             ws_boxes = sorted(ws_results[0].boxes.data.tolist(), key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
            
#             for ws_box in ws_boxes[:1]: # Process only the best windshield candidate
#                 wx1, wy1, wx2, wy2, w_conf, _ = ws_box
#                 wx1, wy1, wx2, wy2 = map(int, [wx1, wy1, wx2, wy2])

#                 # Draw Windshield Box (Green) - Map coords back to original frame
#                 # Offset is (vx1, vy1)
#                 self.draw_box(frame_display, (wx1, wy1, wx2, wy2), "Windshield", (0, 255, 0), offset=(vx1, vy1))

#                 # Crop Windshield
#                 windshield_crop = vehicle_crop[wy1:wy2, wx1:wx2]
#                 if windshield_crop.size == 0: continue

#                 # --- Split Windshield ---
#                 left_img, right_img, mid_point = self.split_image(windshield_crop)

#                 # --- LAYER 3: Seatbelt Detection (on Left & Right halves) ---
                
#                 # 3a. Left Side (Driver usually)
#                 left_results = self.model_seatbelt.predict(left_img, verbose=False)
#                 for sb in left_results[0].boxes.data.tolist():
#                     sx1, sy1, sx2, sy2, _, _ = sb
#                     # Offset logic: Vehicle(vx1, vy1) + Windshield(wx1, wy1) + LeftCrop(0,0)
#                     total_offset_x = vx1 + wx1
#                     total_offset_y = vy1 + wy1
#                     self.draw_box(frame_display, (sx1, sy1, sx2, sy2), "Seatbelt", (0, 0, 255), offset=(total_offset_x, total_offset_y))

#                 # 3b. Right Side (Passenger usually)
#                 right_results = self.model_seatbelt.predict(right_img, verbose=False)
#                 for sb in right_results[0].boxes.data.tolist():
#                     sx1, sy1, sx2, sy2, _, _ = sb
#                     # Offset logic: Vehicle(vx1, vy1) + Windshield(wx1, wy1) + RightCrop(mid_point, 0)
#                     total_offset_x = vx1 + wx1 + mid_point
#                     total_offset_y = vy1 + wy1
#                     self.draw_box(frame_display, (sx1, sy1, sx2, sy2), "Seatbelt", (0, 0, 255), offset=(total_offset_x, total_offset_y))

#         return frame_display

# def run_pipeline(source, weights_dict):
#     """
#     Handles Image, Video, and Webcam sources.
#     source: path to file or '0' for webcam
#     """
#     pipeline = SeatbeltPipeline(
#         weights_dict['vehicle'], 
#         weights_dict['windshield'], 
#         weights_dict['seatbelt']
#     )

#     # Determine source type
#     is_webcam = source == '0' or source == 0
#     if is_webcam:
#         cap = cv2.VideoCapture(0)
#         source_name = "Webcam"
#     elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#         # IMAGE MODE
#         img = cv2.imread(source)
#         if img is None:
#             print("Error: Could not load image.")
#             return
#         result_img = pipeline.process_frame(img)
        
#         # Save or Show
#         cv2.imwrite("output_detected.jpg", result_img)
#         print("Processed image saved as 'output_detected.jpg'")
        
#         # For Colab/Jupyter, use matplotlib to show. For local, use cv2.imshow
#         try:
#             from google.colab.patches import cv2_imshow
#             cv2_imshow(result_img)
#         except ImportError:
#             cv2.imshow("Result", result_img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         return
#     else:
#         # VIDEO MODE
#         cap = cv2.VideoCapture(source)
#         source_name = source

#     # Video/Webcam Loop
#     if not cap.isOpened():
#         print(f"Error: Could not open video source {source}")
#         return

#     # Setup Video Writer (optional, for saving output)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     print(f"Processing {source_name}... Press 'q' to stop.")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Process the frame
#         processed_frame = pipeline.process_frame(frame)

#         # Write to file
#         out.write(processed_frame)

#         # Show result
#         # Note: In Google Colab, cv2_imshow works but is slow for video loops.
#         # If running locally, use cv2.imshow
#         try:
#             from google.colab.patches import cv2_imshow
#             # cv2_imshow(processed_frame) # Uncommenting this in Colab will be very slow
#         except ImportError:
#             cv2.imshow('Seatbelt Detection Pipeline', processed_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
    
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print("Processing complete. Video saved as 'output_video.mp4'")

# # --- CONFIGURATION ---

# # 1. Generic YOLO model for Vehicle Detection (download automatically if not present)
# vehicle_weights_path = "yolov8n.pt" 

# # 2. Your Custom Weights
# windshield_weights_path = "/content/drive/MyDrive/SeatBelt_project/Weights/windsheildWbest.pt"
# seatbelt_weights_path = "/content/drive/MyDrive/SeatBelt_project/Weights/Seat_Belt_weights_dwconv.pt"

# weights_config = {
#     'vehicle': vehicle_weights_path,
#     'windshield': windshield_weights_path,
#     'seatbelt': seatbelt_weights_path
# }

# # --- SELECT SOURCE HERE ---
# # Option A: Image Path
# input_source = "/content/drive/MyDrive/SeatBelt_project/DATA/ALL_Data/Car__018.jpg"

# # Option B: Video Path
# # input_source = "/content/drive/MyDrive/SeatBelt_project/DATA/video.mp4"

# # Option C: Webcam / Real-time
# # input_source = 0 

# # Run
# run_pipeline(input_source, weights_config)