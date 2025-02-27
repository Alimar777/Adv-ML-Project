import torch
import time
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

def main():
    # Check CUDA availability
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Count:", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))
        # Move model to GPU
        device = "cuda"
    else:
        device = "cpu"
        print("Running on CPU")

    # Load BLIP model and processor
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    print("Model loaded successfully")

    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam initialized successfully")
    print("Starting scene description every 10 seconds. Press 'q' to quit.")
    
    last_capture_time = time.time() - 10  # Ensure first capture happens immediately
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Display the frame
            cv2.imshow('Webcam', frame)
            
            current_time = time.time()
            
            # Generate caption every 10 seconds
            if current_time - last_capture_time >= 10:
                # Convert the frame to RGB (from BGR)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # Generate caption
                print("\nGenerating description...")
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
                
                with torch.no_grad():  # No need to track gradients for inference
                    caption_ids = model.generate(**inputs, max_new_tokens=50)
                    
                caption = processor.decode(caption_ids[0], skip_special_tokens=True)
                
                # Display time and caption
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{timestamp}] Scene description: {caption}")
                
                last_capture_time = current_time
            
            # Check if user pressed 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("\nResource released. Program terminated.")

if __name__ == "__main__":
    main()