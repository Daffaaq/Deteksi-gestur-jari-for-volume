import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inisialisasi Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi Pycaw untuk kontrol volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Mendapatkan rentang volume
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    
    # Ibu jari
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # 4 jari lainnya
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers.count(1)

# Membuka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Mengubah warna gambar menjadi RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Menghitung jumlah jari yang terangkat
            fingers_count = count_fingers(hand_landmarks)
            
            if fingers_count == 5:
                volume.SetMasterVolumeLevel(max_vol, None)
                volume_level = 100
            elif fingers_count == 4:
                volume.SetMasterVolumeLevel(min_vol, None)
                volume_level = 0
            else:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                h, w, _ = image.shape
                thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                
                distance = calculate_distance(thumb_tip_coords[0], thumb_tip_coords[1], index_finger_tip_coords[0], index_finger_tip_coords[1])
                vol = np.interp(distance, [30, 200], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)
                volume_level = int(np.interp(vol, [min_vol, max_vol], [0, 100]))
                
            # Menampilkan jumlah jari yang terangkat dan level volume di layar
            cv2.putText(image, f'Fingers: {fingers_count}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Volume: {volume_level}%', 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Volume Control', image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('o'):
        break

cap.release()
cv2.destroyAllWindows()
