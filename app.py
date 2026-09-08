import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- Load Model & Configuration ---
# Place asl_classifier_model.h5 (downloaded from Drive) in the same folder as this file.
keras_model = load_model('asl_classifier_model.h5') 

# IMPORTANT: this must exactly match the `class_names` list printed by the training notebook's
# load_data() call — it's built from sorted(os.listdir(...)), which puts A-Z (uppercase) before
# the lowercase folder names. Double-check against what your notebook actually printed and paste
# it here if it differs even slightly (e.g. dataset variant without "nothing"/"del").
KERAS_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                  'del', 'nothing', 'space']
KERAS_INPUT_SHAPE = (64, 64)  # must match IMAGE_SIZE used during training

# --- MediaPipe Hands setup ---
# static_image_mode=False turns on tracking between frames, which is much faster for video
# than re-running full detection on every single frame.
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MARGIN = 20  # pixels of padding around the detected hand before cropping


def classify_crop(crop_image_rgb):
    """Preprocess an RGB hand crop and run Keras classification.
    Expects RGB input (not BGR) — the training pipeline read images as RGB via
    matplotlib.pyplot.imread(), so inference has to match that color order."""
    resized = cv2.resize(crop_image_rgb, KERAS_INPUT_SHAPE)
    image_array = np.expand_dims(resized, axis=0).astype('float32') / 255.0

    predictions = keras_model.predict(image_array, verbose=0)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])

    return predicted_class, confidence


def gen_frames():
    """Video streaming generator function using OpenCV webcam feed."""
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # OpenCV gives BGR; MediaPipe (and the training data) expect RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        annotated_frame = frame.copy()  # stays BGR — correct for cv2 drawing + JPEG encoding
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]

                x1 = max(int(min(x_coords)) - MARGIN, 0)
                y1 = max(int(min(y_coords)) - MARGIN, 0)
                x2 = min(int(max(x_coords)) + MARGIN, w)
                y2 = min(int(max(y_coords)) + MARGIN, h)

                cropped_hand_rgb = rgb_frame[y1:y2, x1:x2]

                box_color = (255, 255, 255)  # BGR white until classified
                keras_label = ""

                if cropped_hand_rgb.size > 0:
                    try:
                        predicted_class, confidence = classify_crop(cropped_hand_rgb)
                        label_name = KERAS_CLASSES[predicted_class]
                        keras_label = f"{label_name} ({confidence:.2f})"
                        box_color = (0, 255, 0)  # green once we have a prediction
                    except Exception:
                        pass

                text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                if keras_label:
                    cv2.putText(annotated_frame, keras_label, (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
