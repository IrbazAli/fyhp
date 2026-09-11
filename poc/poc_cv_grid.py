import cv2
import numpy as np
import os

def extract_grid_hough(image_path, output_path):
    print("==================================================")
    print("POC 2: Dynamic Grid Extraction (Hough Transform)")
    print("==================================================")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    img = cv2.imread(image_path)
    overlay = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    
    vert_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horiz_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    vert_lines = cv2.HoughLinesP(vert_mask, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
    
    if vert_lines is not None:
        for line in vert_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    horiz_lines = cv2.HoughLinesP(horiz_mask, 1, np.pi/180, threshold=40, minLineLength=100, maxLineGap=150)
    
    if horiz_lines is not None:
        for line in horiz_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(output_path, overlay)
    
    print(f"Dynamic Grid detection complete. Output saved to: {output_path}")
    print("This method uses HoughLinesP to bridge gaps where handwriting intersects the lines.")

if __name__ == "__main__":
    image_file = "../urdu ledtgre.jpeg"
    output_file = "grid_detected.jpg"
    extract_grid_hough(image_file, output_file)
