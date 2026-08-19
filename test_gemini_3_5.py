import os
import google.generativeai as genai
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

try:
    print("Initializing gemini-3.5-flash...")
    vision_model = genai.GenerativeModel('gemini-3.5-flash')
    
    print("Creating dummy image...")
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    
    print("Sending request to Gemini...")
    response = vision_model.generate_content(["What is this?", img])
    
    print(f"SUCCESS! Response: {response.text}")
except Exception as e:
    print(f"FAILED: {e}")
