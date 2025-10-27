import cv2
from ultralytics import YOLO
import time
from collections import deque

# Carica il modello YOLO
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

MAX_PERSONE = 10
SAMPLE_DURATION = 1.0  # Durata campionamento in secondi
FPS_TARGET = 30  # Rilevamenti al secondo

# Buffer per raccogliere i conteggi
conteggi_buffer = deque()
ultimo_aggiornamento = time.time()
conteggio_stabile = 0

# Variabili per controllare il frame rate
ultimo_frame_time = time.time()
frame_interval = 1.0 / FPS_TARGET

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Controlla se è il momento di processare un nuovo frame
    tempo_corrente = time.time()
    if tempo_corrente - ultimo_frame_time < frame_interval:
        # Mostra l'ultimo frame processato senza rilevare
        cv2.imshow('Contatore Persone', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    ultimo_frame_time = tempo_corrente

    # Rileva oggetti nell'immagine con confidence più alta
    results = model(image, classes=[0], verbose=False, conf=0.6)  # class 0 = person
    
    # Conta le persone rilevate
    num_persone_corrente = len(results[0].boxes)
    
    # Aggiungi il conteggio al buffer con timestamp
    conteggi_buffer.append({
        'count': num_persone_corrente,
        'time': tempo_corrente
    })
    
    # Rimuovi conteggi più vecchi di SAMPLE_DURATION
    while conteggi_buffer and (tempo_corrente - conteggi_buffer[0]['time']) > SAMPLE_DURATION:
        conteggi_buffer.popleft()
    
    # Calcola il conteggio stabile (valore massimo negli ultimi 2 secondi)
    if conteggi_buffer:
        conteggio_stabile = max(item['count'] for item in conteggi_buffer)
    
    # Aggiorna display ogni 2 secondi
    if tempo_corrente - ultimo_aggiornamento >= SAMPLE_DURATION:
        print(f"Conteggio stabilizzato: {conteggio_stabile} persone (max degli ultimi {SAMPLE_DURATION}s)")
        ultimo_aggiornamento = tempo_corrente
    
    # Disegna i box intorno alle persone CORRENTI
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        
        # Disegna rettangolo
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{confidence:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Mostra il conteggio STABILE (non quello corrente)
    if conteggio_stabile > MAX_PERSONE:
        cv2.putText(image, 'LIMITE MASSIMO SUPERATO!', (50, 50), 
                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(image, f'Persone: {conteggio_stabile}/{MAX_PERSONE}', (50, 100), 
                   cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
    else:
        cv2.putText(image, f'Persone: {conteggio_stabile}/{MAX_PERSONE}', (70, 100), 
                   cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 4)
    
    # Mostra info sul conteggio corrente in piccolo
    cv2.putText(image, f'Rilevamento: {num_persone_corrente}', (70, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Mostra quanti campioni nel buffer
    cv2.putText(image, f'Campioni: {len(conteggi_buffer)}', (70, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow('Contatore Persone', image)
    
    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()