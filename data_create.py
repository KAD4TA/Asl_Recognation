import cv2
import mediapipe as mp
import math
import time
import numpy as np
# start hand recognition by mediapipe modul
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
imgSize=400
image_count=0
PATH='my_DATA/validation/R'
# start the camera
cap = cv2.VideoCapture(0)

while True:
    # capture image
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # convert image from bgr to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # make hand recognition
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Use hand positions to draw the frame
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y


            x_min -= 30
            x_max += 30
            y_min -= 30
            y_max += 30
            
            # Draw a rectangle around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            imgWhite= np.ones((imgSize,imgSize,3),dtype=np.uint8)*255
            imgCrop= frame[y_min:y_max, x_min:x_max]
            imgCropShape= imgCrop.shape

        aspectRatio= y_max / x_max

        if aspectRatio > 1:                         # image stretching by width and length
            k = imgSize / y_max
            wCalculate= math.ceil(k*x_max)
            imgResize= cv2.resize(imgCrop, (wCalculate,imgSize))
            imgResizeShape= imgResize.shape
            wGap= math.ceil((imgSize-wCalculate) / 2)
            imgWhite[:, wGap:wCalculate + wGap] = imgResize

        else:
            k = imgSize / x_max
            hCalculate= math.ceil(k*y_max)
            imgResize= cv2.resize(imgCrop,(imgSize,hCalculate))
            imgResizeShape= imgResize.shape
            hGap= math.ceil((imgSize-hCalculate) / 2)
            imgWhite[hGap:hCalculate + hGap, :] = imgResize
            
        cv2.imshow('whiteBackground',imgWhite)

    if cv2.waitKey(1) == ord('b'):                 # press the b button save image
        if image_count<500:
            cv2.imwrite(f'{PATH}/images_{time.time()}.jpg', imgWhite)
            image_count+=1 
            print(str(image_count)+ '\tsaved image')
            
    
    # show the frame
    cv2.imshow('Hand Rectangle', frame)
    
    # press the q button to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera
cap.release()
cv2.destroyAllWindows()

