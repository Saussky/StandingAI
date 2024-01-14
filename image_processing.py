import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

def classify_images(model, directory):
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, file_name)
            img = load_and_preprocess_image(file_path)

            # Predict
            prediction = model.predict(img)
            classification = 'Standing' if prediction[0][0] > 0.5 else 'Sitting'

            # Print the result
            print(f"Image: {file_name} | Prediction: {classification} ({prediction[0][0]})")

def classify_image(model, image_path):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    print('prediction is: ', prediction)
    classification = 'Standing' if prediction[0][0] > 0.5 else 'Sitting'
    return classification

def capture_image():
    cap = cv2.VideoCapture(0)
    success, image = cap.read()
    if success:
        # Save or process your image here
        cv2.imwrite('current_frame.jpg', image)
    cap.release()
    cv2.destroyAllWindows()
