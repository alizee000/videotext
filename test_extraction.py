import cv2
import time
import numpy as np

def extract_keyframes(video_path: str, diff_threshold: float = 0.01):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        cap.release()
        return frames
        
    frames.append(prev_frame)
    
    h, w = prev_frame.shape[:2]
    math_w = 240
    math_h = int((math_w / float(w)) * h)
    
    prev_math_img = cv2.resize(prev_frame, (math_w, math_h))
    prev_gray = cv2.cvtColor(prev_math_img, cv2.COLOR_BGR2GRAY)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    step = max(1, int(fps / 5)) 
    
    frame_count = 1
    while True:
        ret = cap.grab()
        if not ret:
            break
            
        if frame_count % step == 0:
            ret, curr_frame = cap.retrieve()
            if not ret or curr_frame is None:
                continue
                
            curr_math_img = cv2.resize(curr_frame, (math_w, math_h))
            curr_gray = cv2.cvtColor(curr_math_img, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(curr_gray, prev_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            changed_pixels = cv2.countNonZero(thresh)
            total_pixels = math_w * math_h
            
            percentage_changed = changed_pixels / total_pixels
            if percentage_changed > diff_threshold:
                frames.append(curr_frame)
                prev_gray = curr_gray
                
        frame_count += 1
            
    cap.release()
    return frames

start = time.time()
f = extract_keyframes("front_product.mp4")
print(f"Extraction took {time.time()-start:.2f}s")
print(f"Extracted {len(f)} frames")
