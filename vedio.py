import cv2
import re
from ultralytics import YOLO
import easyocr

# 1. Initialize
model = YOLO('best.pt') 
# Use gpu=True if you have CUDA set up, otherwise gpu=False
reader = easyocr.Reader(['en'], gpu=False) 

cap = cv2.VideoCapture('new11.mp4')

# final_verified_plates stores the "Best Result" (Highest Accuracy)
final_verified_plates = {} 
frame_count = 0 
process_every_n_frames = 5  

# active_scan_results stores the "Live Reading" for the video overlay
active_scan_results = [] 

def clean_text(t):
    return re.sub(r'[^A-Z0-9]', '', t.upper())

print("--- ANPR System Live: Monitoring for High Accuracy Plates ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    
    # --- AI PROCESSING BLOCK (Live Reading) ---
    if frame_count % process_every_n_frames == 0:
        results = model.track(frame, persist=True, conf=0.4, verbose=False)[0] 
        
        new_scan_data = []

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                if plate_crop.size > 0:
                    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    ocr_results = reader.readtext(gray_plate)
                    
                    for (_, text, ocr_conf) in ocr_results:
                        clean_p = clean_text(text)
                        
                        # --- LIVE FILTER (Show anything > 70% on screen, keep > 86% for results) ---
                        if len(clean_p) >= 5:
                            # Update the LIVE overlay data
                            new_scan_data.append({
                                'coords': [int(x1), int(y1), int(x2), int(y2)],
                                'text': clean_p,
                                'live_conf': ocr_conf
                            })
                            
                            # Update the FINAL RESULT only if accuracy is > 86% AND it's a new record high
                            if ocr_conf > 0.86:
                                if clean_p not in final_verified_plates or ocr_conf > final_verified_plates[clean_p]:
                                    final_verified_plates[clean_p] = ocr_conf
                                    print(f"[NEW BEST RESULT] Plate: {clean_p} | Accuracy: {ocr_conf*100:.1f}%")
        
        # Update display data
        if new_scan_data:
            active_scan_results = new_scan_data
        elif frame_count % 20 == 0: 
            active_scan_results = []

    # --- RENDERING BLOCK (Drawing the Live Reading) ---
    for item in active_scan_results:
        c = item['coords']
        # Green box for detection
        cv2.rectangle(frame, (c[0], c[1]), (c[2], c[3]), (0, 255, 0), 2)
        
        # Background for live label
        cv2.rectangle(frame, (c[0], c[1] - 35), (c[0] + 350, c[1]), (0, 255, 0), -1)
        
        # Display LIVE READING on the video frame
        live_label = f"LIVE: {item['text']} ({item['live_conf']*100:.1f}%)"
        cv2.putText(frame, live_label, (c[0] + 5, c[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 3. VISUAL FEEDBACK
    cv2.imshow('ANPR - Live Reading vs Best Result', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# Final Summary Table (Shows the Final Verified Results)
print("\n" + "="*45)
print("     FINAL VERIFIED RESULTS (BEST SCANS)     ")
print("="*45)
if not final_verified_plates:
    print("No plates met the 86% verification threshold.")
else:
    sorted_plates = sorted(final_verified_plates.items(), key=lambda x: x[1], reverse=True)
    for i, (plate, conf) in enumerate(sorted_plates, 1):
        print(f"{i}. Plate: {plate} | Highest Accuracy: {conf*100:.1f}%")
print("="*45)