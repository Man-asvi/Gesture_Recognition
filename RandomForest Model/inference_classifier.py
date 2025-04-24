import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except FileNotFoundError:
    print("❌ Error: Model file 'model.p' not found.")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary for labels
labels_dict = {0: 'Peace', 1: 'Hi', 2: 'A'}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.75  # You can adjust this based on testing

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Use only the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]

        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            x_.append(x)
            y_.append(y)

        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        # Get bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Predict probabilities
        probabilities = model.predict_proba([np.asarray(data_aux)])[0]
        confidence = np.max(probabilities)
        predicted_class = np.argmax(probabilities)

        if confidence > CONFIDENCE_THRESHOLD:
            predicted_character = labels_dict[int(predicted_class)]
        else:
            predicted_character = 'Unknown'

        # Draw rectangle and put label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()