import os
import cv2
import tempfile
import time
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PySceneDetect removed in favor of custom UI pixel diffing

app = FastAPI(title="Video to User Story API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Initialization ---
# Setup Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set. Please add it to the .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model for multimodal video processing
vision_model = genai.GenerativeModel('gemini-3.5-flash')

# --- Helper Functions ---

def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if an image is blurry using the variance of the Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def save_frames_to_disk(frames: list, output_dir: str):
    """Background task to save high-resolution frames to disk without blocking API response."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{output_dir}/frame_{i:04d}.jpg", frame)

def extract_keyframes(video_path: str, diff_threshold: float = 0.01):
    """
    Custom extraction algorithm to find the optimal number of frames without data loss.
    Uses pixel difference to only extract frames when the UI state significantly changes.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return frames
        
    frames.append(prev_frame)
    
    # Calculate resize dimensions for 240px width math (preserve aspect ratio)
    h, w = prev_frame.shape[:2]
    math_w = 240
    math_h = int((math_w / float(w)) * h)
    
    prev_math_img = cv2.resize(prev_frame, (math_w, math_h))
    prev_gray = cv2.cvtColor(prev_math_img, cv2.COLOR_BGR2GRAY)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        
    # Sample roughly every 0.2 seconds to catch fast clicks
    step = max(1, int(fps / 5)) 
    
    frame_count = 1
    while True:
        # Sequential grab is exponentially faster than full decoding
        ret = cap.grab()
        if not ret:
            break
            
        if frame_count % step == 0:
            ret, curr_frame = cap.retrieve()
            if not ret:
                continue
                
            curr_math_img = cv2.resize(curr_frame, (math_w, math_h))
            curr_gray = cv2.cvtColor(curr_math_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute pixel difference on the tiny thumbnail
            diff = cv2.absdiff(curr_gray, prev_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            changed_pixels = cv2.countNonZero(thresh)
            total_pixels = math_w * math_h
            
            percentage_changed = changed_pixels / total_pixels
            
            # If more than 1% of the screen changed, capture it as a new state!
            if percentage_changed > diff_threshold:
                frames.append(curr_frame)
                prev_gray = curr_gray
                
        frame_count += 1
            
    cap.release()
    
    # Ensure we don't exceed API limits for extremely long/active videos
    if len(frames) > 60:
        step_down = len(frames) / 60.0
        reduced_frames = [frames[int(i * step_down)] for i in range(60)]
        return reduced_frames
        
    return frames

def generate_narrative_from_sequence(pil_images: list) -> str:
    """
    Send the entire sequence of images to Gemini 3.5 Flash for deep contextual UI analysis.
    """
    if not GEMINI_API_KEY:
        return '{"error": "Missing Gemini API Key"}'
        
    prompt = '''
You are an expert Agile Product Owner and UX Researcher.
I am providing you with a chronological sequence of frames extracted from a UI screen recording video.

CRITICAL INSTRUCTIONS:
1. DO NOT HALLUCINATE. You must ONLY describe exactly what is visibly present in the frames. Analyze the chronological actions taken by the user.
2. Group related actions into logical features or tasks. For each distinct feature or workflow, generate a separate JIRA User Story.
3. You MUST output your response as a strictly valid JSON array of objects exactly like this:
[
  {
    "title": "A concise, clear title for the JIRA ticket",
    "description": "As a [User Persona], \\nI want to [Action/Goal], \\nSo that [Value/Benefit].\\n\\n*Acceptance Criteria:*\\n- [Criterion 1]\\n- [Criterion 2]\\n\\n*Workflow Observed:*\\n1. [Step 1]\\n2. [Step 2]"
  }
]
'''
    
    # The payload is the prompt followed by all the images in chronological order
    payload = [prompt] + pil_images
    
    try:
        response = vision_model.generate_content(
            payload,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini sequence processing failed: {e}")
        return '{"error": "Gemini processing failed"}'

# --- API Endpoints ---

@app.post("/api/v1/analyze-video")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a video.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        content = await file.read()
        tmp_video.write(content)
        tmp_video_path = tmp_video.name
        
    try:
        print("Extracting keyframes...")
        keyframes = extract_keyframes(tmp_video_path)
        
        stats = {"total_frames": len(keyframes), "clear_frames": 0, "blurry_frames": 0}
        
        session_id = int(time.time())
        output_dir = f"extracted_frames/session_{session_id}"
        
        # Filter blurry frames and compress payload
        clear_pil_images = []
        frames_to_save = []
        
        print(f"Filtering {len(keyframes)} keyframes...")
        for i, frame in enumerate(keyframes):
            # Payload Compression: Resize to 768px max dimension for fast network upload
            h, w = frame.shape[:2]
            max_dim = 768.0
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                payload_frame = cv2.resize(frame, (new_w, new_h))
            else:
                payload_frame = frame
                
            # Run blur check on the mathematically smaller frame!
            if not is_blurry(payload_frame):
                # Queue original high-res frame for background disk save
                frames_to_save.append(frame)
                
                # Convert OpenCV BGR to RGB PIL Image
                rgb_image = cv2.cvtColor(payload_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                clear_pil_images.append(pil_image)
                stats["clear_frames"] += 1
            else:
                stats["blurry_frames"] += 1
                
        # Queue the disk saving to happen in the background after the API returns!
        background_tasks.add_task(save_frames_to_disk, frames_to_save, output_dir)
        
        print(f"Sending {len(clear_pil_images)} clear frames to Gemini 3.5 Flash for UX analysis...")
        raw_response = generate_narrative_from_sequence(clear_pil_images)
        
        try:
            # Clean possible markdown wrapping from LLM response
            clean_json = raw_response.replace("```json", "").replace("```", "").strip()
            jira_tickets = json.loads(clean_json)
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            jira_tickets = []
        
        return JSONResponse(content={
            "status": "success",
            "jira_tickets": jira_tickets,
            "processing_stats": stats
        })
        
    except Exception as e:
        print(f"Internal Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)

# --- Serve Static UI ---
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
