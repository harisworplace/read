import cv2
import os
from ultralytics import YOLO
import easyocr

# Initialize Models
model = YOLO('best.pt') 
reader = easyocr.Reader(['en'])

img_path = 'car2.jpeg'

# Print Header (as seen in your screenshot)
print("==========================================")
print("     AI LICENSE PLATE RECOGNITION SYSTEM")
print("     Global Model Accuracy (mAP@50): 87.61%")
print("==========================================\n")

if os.path.exists(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    max_width = 800
    img = cv2.resize(img, (max_width, int(h * (max_width/w))), interpolation=cv2.INTER_AREA)

    # Detection
    results = model(img)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        # Crop and process for OCR
        plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Recognition
        ocr_result = reader.readtext(thresh_plate)
        
        if ocr_result:
            plate_text = " ".join([res[1] for res in ocr_result]).upper()
            
            # --- Modified Print Block to match your screenshot ---
            print("[NEW DETECTION]")
            print(f"Detected Text: {plate_text}")
            # Format the confidence score to 2 decimal places
            print(f"YOLO Detection Accuracy: {score * 100:.2f}%")
            print("------------------------------------------")
            # -----------------------------------------------------

            # Visual Feedback
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"PLATE: {plate_text}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('ANPR Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Error: File '{img_path}' not found.")