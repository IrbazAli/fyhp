import cv2
import pytesseract
import sys
import os

# Note: You may need to set the tesseract_cmd path if it's not in your system PATH
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_ocr_failure(image_path):
    print("==================================================")
    print("POC 1: Testing Standard OCR on Nastaliq Khata")
    print("==================================================")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    # Try running Tesseract configured for Urdu
    print("Running Tesseract OCR (lang='urd')...")
    try:
        # We use lang='urd' for Urdu. If not installed, this will throw an error.
        text = pytesseract.image_to_string(image, lang='urd')
        
        print("\n--- OCR OUTPUT START ---")
        print(text)
        print("--- OCR OUTPUT END ---\n")
        
        print("Conclusion: Standard OCR (like Tesseract) fails to produce meaningful")
        print("results on cursive, handwritten Nastaliq on noisy ledger paper.")
        print("This proves the necessity of a custom HTR (CRNN/TrOCR) pipeline.")
    except Exception as e:
        print(f"\nError running OCR: {e}")
        print("Make sure Tesseract is installed and the 'urd' language pack is downloaded.")

if __name__ == "__main__":
    # Adjust this path if your image is located elsewhere
    image_file = "../urdu ledtgre.jpeg"
    test_ocr_failure(image_file)
