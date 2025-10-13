import cv2
from ultralytics import YOLO

# Carica il modello YOLO
model = YOLO('yolov8n.pt')  # Scarica automaticamente il modello

cap = cv2.VideoCapture(0)

MAX_PERSONE = 3

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Rileva oggetti nell'immagine
    results = model(image, classes=[0], verbose=False)  # class 0 = person
    
    # Conta le persone rilevate
    num_persone = len(results[0].boxes)
    
    # Disegna i box intorno alle persone
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        
        # Disegna rettangolo
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{confidence:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Controlla se il limite è superato
    if num_persone > MAX_PERSONE:
        cv2.putText(image, 'LIMITE MASSIMO SUPERATO!', (50, 50), 
                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(image, f'Persone: {num_persone}/{MAX_PERSONE}', (50, 100), 
                   cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
        print(f"ATTENZIONE! Persone rilevate: {num_persone} - LIMITE SUPERATO!")
    else:
        cv2.putText(image, f'Persone: {num_persone}/{MAX_PERSONE}', (70, 100), 
                   cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 4)
        print(f"Persone rilevate: {num_persone}")

    cv2.imshow('Contatore Persone', image)
    
    # Premi 'q' per uscire
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()