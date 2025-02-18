import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import requests
import os

# Model Architecture
def build_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(48, 48, 1)),  # 48x48 grayscale images
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # Safe (0) and Threat (1)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training Function
def train_model(X_train, y_train, X_val, y_val):
    model = build_model()

    # Normalize the data
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    # Save the trained model
    model.save('face_setnewer_model.h5')
    return model

# Face Detection and Classification Function
def detect_and_classify_face(model, frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
        face_roi_gray = face_roi_gray.astype("float32") / 255.0
        face_roi_gray = np.expand_dims(face_roi_gray, axis=-1)

        # Predict face class using the model
        prediction = model.predict(np.expand_dims(face_roi_gray, axis=0))
        predicted_class = np.argmax(prediction, axis=1)

        # Display the label on the screen
        label = "Safe" if predicted_class == 0 else "Threat"
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# Query human behavior and safety via external API
def query_human_behavior(frame):
    """
    Sends human face data to the external API to get behavioral analysis and safety status.
    """
    # Encode frame to send via POST request
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Send the image data to the external API
    response = requests.post("https://azencompileropensourcefoundation.com/visual-enginnering/realtime.php", files={'image': img_bytes})

    if response.status_code == 200:
        try:
            # Clean the response text to handle extra data
            clean_response = response.text.split('\n')[0]  # Assuming extra data is in subsequent lines
            print(clean_response)  # Print the cleaned response for debugging
            return json.loads(clean_response)  # Parse cleaned response as JSON
        except ValueError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("Failed to query behavior!")
        return None

# Query the human behavior status for safety and threats
def analyze_and_respond_to_behavior(human_data):
    """
    Analyze the human behavior based on the data received from the external API and provide a response.
    """
    if human_data.get("movement") == "aggressive":
        print("Response: Assume defensive posture!")
    elif human_data.get("movement") == "fearful":
        print("Response: Prepare to evacuate!")
    else:
        print("Response: Normal behavior detected.")

# Function to get graphical view data (replacing camera feed)
def get_graphical_view():
    response = requests.get("https://azencompileropensourcefoundation.com/visual-enginnering/photo-video-object.php")
    if response.status_code == 200:
        return np.array(bytearray(response.content), dtype=np.uint8)
    else:
        print("Failed to retrieve graphical view!")
        return None

# Save model function
def save_model(model):
    model.save('face_setnewer_model.h5')
    print("Model saved successfully!")

# Function to check if camera is operational
def is_camera_working():
    frame_data = get_graphical_view()
    return frame_data is not None

# Main Function
if __name__ == "__main__":
    try:
        model = None
        if os.path.exists('face_setnewer_model.h5'):
            model = tf.keras.models.load_model('face_setnewer_model.h5')  # Load pre-trained model
            print("Model loaded successfully.")
        else:
            print("Model not found! Building a new one.")
            model = build_model()
            model.save('face_setnewer_model.h5')  # Save a newly built model
            print("Model trained and saved.")

        # Automatically save the model regardless of the camera feed or any error
        save_model(model)

        while True:
            if not is_camera_working():
                print("Error: No graphical view retrieved. Proceeding with backup.")
                # Here you can implement fallback logic for when camera is not working
                break

            frame_data = get_graphical_view()

            if frame_data is None:
                print("Error: No graphical view retrieved.")
                break

            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode the image!")
                break
            
            # Query the human behavior and safety from the external source
            human_data = query_human_behavior(frame)
            
            if human_data:
                # Analyze the human's behavior
                analyze_and_respond_to_behavior(human_data)
            
            # Detect and classify the face using the previously trained model
            frame = detect_and_classify_face(model, frame)

            # Show the frame
            cv2.imshow('Graphical View', frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        if model:
            save_model(model)  # Save model in case of error
