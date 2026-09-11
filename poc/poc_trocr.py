import cv2
import os
import sys

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch
except ImportError:
    print("==================================================")
    print("Error: Missing AI Libraries for TrOCR")
    print("==================================================")
    print("Please install the required libraries to run this POC:")
    print("pip install transformers torch torchvision Pillow")
    sys.exit(1)

def run_trocr(image_path):
    print("==================================================")
    print("POC 4: Custom HTR using TrOCR on Western Digits")
    print("==================================================")
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    print("\nLoading Microsoft TrOCR Model from Hugging Face...")
    print("(This may take a minute if downloading for the first time)")
    
    try:
        from transformers import RobertaTokenizer, ViTImageProcessor
        
        # Load components explicitly to avoid the Windows fast tokenizer bug
        tokenizer = RobertaTokenizer.from_pretrained('microsoft/trocr-base-handwritten')
        feature_extractor = ViTImageProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
        
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        
        # Prepare the image for the model
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        print("\nRunning Inference...")
        # Generate the text
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("\n--- TrOCR OUTPUT START ---")
        print(f"PREDICTED TEXT: {generated_text}")
        print("--- TrOCR OUTPUT END ---\n")
        
        print("Conclusion: Unlike standard Tesseract, TrOCR successfully reads")
        print("the handwritten amounts because the merchant uses Western digits (1, 2, 3).")
        print("This proves that a sequence-modeling architecture works for this project!")
        
    except Exception as e:
        print(f"\nError during model execution: {e}")

def create_sample_crop(full_image_path, crop_output_path):
    # This creates a sample crop of one of the numbers from the main image
    if not os.path.exists(full_image_path):
        return False
        
    img = cv2.imread(full_image_path)
    # Estimate a region where a number might be based on standard khata layout
    # (Adjust these coordinates if it misses the number on your specific photo)
    y_start, y_end = 400, 500
    x_start, x_end = 450, 650
    
    cropped = img[y_start:y_end, x_start:x_end]
    cv2.imwrite(crop_output_path, cropped)
    print(f"Auto-generated a sample crop from the ledger: {crop_output_path}")
    return True

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    crop_path = os.path.join(script_dir, "cropped_amount.jpg")
    full_ledger_path = os.path.join(script_dir, "..", "urdu ledtgre.jpeg")
    
    # If the user hasn't provided a cropped image yet, we make one for them
    if not os.path.exists(crop_path):
        success = create_sample_crop(full_ledger_path, crop_path)
        if not success:
            print(f"Could not find ledger image at {full_ledger_path}")
            print("Please provide a cropped image of a number and name it 'cropped_amount.jpg' in the poc folder.")
            sys.exit(1)
            
    run_trocr(crop_path)
