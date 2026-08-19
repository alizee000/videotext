from transformers import BlipProcessor
import numpy as np
from PIL import Image
import traceback

try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Create dummy image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    
    print("Testing processor without text...")
    inputs = processor(images=pil_image, return_tensors="pt")
    print("Success without text")
    
except Exception as e:
    print("Error without text:")
    traceback.print_exc()

try:
    print("\nTesting processor with text and padding...")
    inputs = processor(images=pil_image, text="a scene showing", return_tensors="pt", padding=True)
    print("Success with text")
except Exception as e:
    print("Error with text:")
    traceback.print_exc()
