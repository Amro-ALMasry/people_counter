# 🧍‍♂️ Conta Persone con YOLOv8

Questo progetto utilizza **OpenCV** e **YOLOv8** per rilevare e contare le persone in tempo reale tramite la webcam del computer.  
Se il numero di persone supera un limite massimo (preimpostato nel codice), sullo schermo viene mostrato un avviso visivo e testuale.

---

## 🎯 Funzionalità principali

- Rileva le **persone** in tempo reale dalla webcam.  
- Disegna un **rettangolo verde** attorno a ogni persona riconosciuta.  
- Mostra il **numero totale** di persone rilevate.  
- Se viene superato un limite massimo (es. 3 persone), mostra un messaggio di **avviso rosso** sullo schermo e nel terminale.

---

## 🧩 Requisiti

Assicurati di avere **Python 3.8 o superiore** installato.  
Poi installa le librerie necessarie con i seguenti comandi:

```bash
pip install ultralytics
pip install opencv-python
