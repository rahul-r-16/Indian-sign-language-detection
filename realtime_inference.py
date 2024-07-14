import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('indian_sign_language_model.h5')

# Define classes (adjust based on your dataset)
classes = ['A', 'B', 'C', ...]

# Function for real-time video processing and detection
def detect_sign_language():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame (resize, normalize, etc.)
        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255.0  # Normalize pixel values
        
        # Make prediction
        pred = model.predict(np.expand_dims(normalized_frame, axis=0))
        pred_class = np.argmax(pred)
        sign = classes[pred_class]
        
        # Display result on frame
        cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Indian Sign Language Detection', frame)
        
        # Exit on ESC key press
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_sign_language()
