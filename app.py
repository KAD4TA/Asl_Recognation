from keras.models import load_model# TensorFlow is required for Keras to work
import tensorflow as tf
import mediapipe as mp
import cv2  # Install opencv-python
import numpy as np
import math
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("My_models\\mobilenet_a_d\\mobile_a_d_predict.h5")
model.load_weights("My_models\\mobilenet_a_d\\mobile_a_d_weights.h5")
model_e_h= load_model("My_models\\mobilenet_e_h\\mobile_e_h_predict.h5")
model_e_h.load_weights("My_models\\mobilenet_e_h\\mobile_e_h_weights.h5")
model_i_m= load_model("My_models\\mobilenet_I_M\\mobile_I_M_predict.h5")
model_i_m.load_weights("My_models\\mobilenet_I_M\\mobile_I_M_weights.h5")
model_n_q= load_model("My_models\\mobilenet_n_q\\mobile_n_q_predict.h5")
model_n_q.load_weights("My_models\\mobilenet_n_q\\mobile_n_q_weights.h5")
model_r_u= load_model("My_models\\mobilenet_r_u\\mobile_r_u_predict.h5")
model_r_u.load_weights("My_models\\mobilenet_r_u\\mobile_r_u_weights.h5")
model_v_y= load_model("My_models\\mobilenet_v_y\\mobile_v_y_predict.h5")
model_v_y.load_weights("My_models\\mobilenet_v_y\\mobile_v_y_weights.h5")
ofset=30
imgSize=600
# Load the labels
class_names = open("Labels/label.txt", "r").readlines()
class_names_e_h= open("Labels/label_e_h.txt","r").readlines()
class_names_i_m= open("Labels/label_I_M.txt","r").readlines()
class_names_n_q= open("Labels/label_n_q.txt","r").readlines()
class_names_r_u= open("Labels/label_r_u.txt","r").readlines()
class_names_v_y= open("Labels/label_v_y.txt","r").readlines()
# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    image_copy= image.copy()
    if not ret:
        print("has a problem")
        break
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # hand recognation
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y


            x_min -= 30
            x_max += 30
            y_min -= 30
            y_max += 30
            
            # draw a rectangle side of hand
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            imgWhite= np.ones((imgSize,imgSize,3),dtype=np.uint8)*255
            imgCrop= image[y_min:y_max, x_min:x_max]
            imgCropShape= imgCrop.shape

        aspectRatio= y_max / x_max

        if aspectRatio > 1:
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
        imageres=cv2.resize(imgWhite,(224,224))
    
    # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(imageres, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
        image = image / 255

    # Predicts the model
        prediction = model.predict(image)
        prediction_e_h= model_e_h.predict(image)
        prediction_i_m= model_i_m.predict(image)
        prediction_n_q= model_n_q.predict(image)
        prediction_r_u= model_r_u.predict(image)
        prediction_v_y= model_v_y.predict(image)
        #----------------------------------------
        index = np.argmax(prediction)
        index_e_h= np.argmax(prediction_e_h)
        index_I_M= np.argmax(prediction_i_m)
        index_n_q= np.argmax(prediction_n_q)
        index_r_u= np.argmax(prediction_r_u)
        index_v_y= np.argmax(prediction_v_y)
        #------------------------------------------
        a_d = np.max(prediction)
        e_h = np.max(prediction_e_h)
        i_m = np.max(prediction_i_m)
        n_q = np.max(prediction_n_q)
        r_u = np.max(prediction_r_u)
        v_y = np.max(prediction_v_y)

        liste=[a_d, e_h, i_m, n_q, r_u, v_y]   # parametres inside the liste 

        max_value = max(liste)
         #  confidence scores
        confidence_score     = prediction[0][index]
        confidence_score_e_h = prediction_e_h[0][index_e_h]
        confidence_score_i_m = prediction_i_m[0][index_I_M]
        confidence_score_n_q = prediction_n_q[0][index_n_q]
        confidence_score_r_u = prediction_r_u[0][index_r_u]
        confidence_score_v_y = prediction_v_y[0][index_v_y]
    
        if  max_value == a_d:                            # The answer is given according to which the max value is.   
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
            cv2.putText(image_copy, f'Predicted: {class_name} ({confidence_score:.2f})', (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif  max_value == e_h:
            class_name_e_h = class_names_e_h[index_e_h]
            confidence_score_e_h = prediction_e_h[0][index_e_h]
            print("Class:", class_name_e_h[2:], end=" ")
            print("Confidence Score:", str(np.round(confidence_score_e_h * 100))[:-2], "%")
            cv2.putText(image_copy, f'Predicted: {class_name_e_h} ({confidence_score_e_h:.2f})', (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif  max_value == i_m:
            class_name_i_m= class_names_i_m[index_I_M]
            confidence_score_I_M = prediction_i_m[0][index_I_M]
            print("Class:", class_name_i_m[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score_I_M * 100))[:-2], "%")
            cv2.putText(image_copy, f'Predicted: {class_name_i_m} ({confidence_score_I_M:.2f})', (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        elif  max_value == n_q:
            class_name_n_q= class_names_n_q[index_n_q]
            confidence_score_n_q = prediction_n_q[0][index_n_q]
            print("Class:", class_name_n_q[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score_n_q * 100))[:-2], "%")
            cv2.putText(image_copy, f'Predicted: {class_name_n_q} ({confidence_score_n_q:.2f})', (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        elif  max_value == r_u:
            class_name_r_u= class_names_r_u[index_r_u]
            confidence_score_r_u = prediction_r_u[0][index_r_u]
            print("Class:", class_name_r_u[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score_r_u * 100))[:-2], "%")
            cv2.putText(image_copy, f'Predicted: {class_name_r_u} ({confidence_score_r_u:.2f})', (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        elif  max_value == v_y:
            class_name_v_y= class_names_v_y[index_v_y]
            confidence_score_v_y = prediction_v_y[0][index_v_y]
            print("Class:", class_name_v_y[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score_v_y * 100))[:-2], "%")
            cv2.putText(image_copy, f'Predicted: {class_name_v_y} ({confidence_score_v_y:.2f})', (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            print("it's getting harder to predict")
            cv2.putText(image_copy, "it's getting harder to predict", (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == ord('q'):
        break
    cv2.imshow('frame',image_copy)
camera.release()
cv2.destroyAllWindows()