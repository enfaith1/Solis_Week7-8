# ASL Sign Language Classifier
> Submitted by: Solis BSCS - 3A

A computer vision project that uses a **Convolutional Neural Network (CNN)** to recognize **American Sign Language (ASL)** hand signs from images and apply the trained model to a **live webcam application**.

The project has two main parts:

1. **Model training (`Solis_WEEK_7_8.ipynb`)** – prepares the ASL dataset, trains and evaluates the CNN, and saves the trained model as `asl_classifier_model.h5`.
2. **Live inference (`app.py`)** – loads the `.h5` model, detects a hand from the webcam using MediaPipe, crops the detected hand, preprocesses it to match the training format, and displays the predicted ASL class with a confidence value through a Flask web page.

> **Important:** The reported **93.1% accuracy is the validation accuracy from the notebook's training run.** It should not be interpreted as a guaranteed real-world/webcam accuracy. Live performance can vary because the webcam pipeline uses MediaPipe to crop the hand before classification.

## Project Overview

The goal of the project is to demonstrate an end-to-end machine learning and computer vision workflow for ASL sign recognition.

The system learns visual patterns from ASL images and then uses those learned patterns to classify a hand sign into one of **29 classes**:

- `A` through `Z`
- `del`
- `nothing`
- `space`

The trained CNN expects an RGB image with a fixed size of **64 × 64 pixels**. During live inference, the application captures frames from the webcam, detects the hand with MediaPipe, creates a bounding box around the hand with a small margin, and sends the resulting RGB crop to the CNN.

## How the System Works

```text
                    MODEL TRAINING

ASL Alphabet Dataset
        |
        v
Copy up to 1,000 images/class
        |
        v
29,000 images total
        |
        v
Resize to 64 x 64 + RGB normalization
        |
        v
80% Training / 20% Validation
        |
        v
CNN + Data Augmentation
        |
        v
Model Evaluation
        |
        v
asl_classifier_model.h5


                    LIVE INFERENCE

Webcam
  |
  v
OpenCV frame capture
  |
  v
MediaPipe hand detection
  |
  v
Hand bounding box + 20 px margin
  |
  v
RGB hand crop
  |
  v
Resize to 64 x 64 + /255 normalization
  |
  v
CNN (.h5 model)
  |
  v
Predicted ASL class + confidence
  |
  v
Flask web interface
```

## Supported ASL Classes

The class order used by the trained model and by `app.py` is:

```text
A, B, C, D, E, F, G, H, I, J, K, L, M,
N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
del, nothing, space
```

The order matters because the neural network outputs a probability for each class by index. `app.py` uses the same order in `KERAS_CLASSES` so the numeric model output is mapped back to the correct ASL label.

## Model Training Notebook — `Solis_WEEK_7_8.ipynb`

The notebook is the main machine learning component of the project. It performs the complete training workflow from dataset preparation to model export.

### 1. Google Drive setup

The notebook mounts Google Drive so the dataset and trained model can persist beyond the temporary Colab runtime.

The training workflow uses Drive for storage and copies the selected dataset to the Colab VM's local disk for faster access during training.

### 2. Dataset

The ASL image dataset used to train this project is based on the **ASL Alphabet** dataset by Grassknoted, available on Kaggle:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

The dataset contains labeled images representing American Sign Language (ASL) hand signs. This project uses the dataset as the basis for training the CNN image classification model and limits the data to a maximum of **1,000 images per class**.


The recorded training run contained:

| Item | Value |
|---|---:|
| Classes | 29 |
| Images per class | 1,000 |
| Total images | 29,000 |
| Image size | 64 × 64 |
| Channels | 3 (RGB) |

The 29 classes are `A-Z`, `del`, `nothing`, and `space`.

### 3. Image preprocessing

Each image is:

- Read as RGB.
- Converted to 3-channel RGB when necessary.
- Resized to **64 × 64 pixels**.
- Converted into an array suitable for TensorFlow/Keras.
- Normalized from the `[0, 255]` pixel range to `[0, 1]` by dividing by 255.

The resulting dataset tensor had the shape:

```text
(29000, 64, 64, 3)
```

### 4. Train/validation split

The notebook uses an **80/20 split** with `random_state=42` for reproducibility.

| Dataset portion | Images |
|---|---:|
| Training | 23,200 |
| Validation | 5,800 |
| Total | 29,000 |

The validation images are kept separate from the training process and are used to evaluate how well the model generalizes to unseen validation examples.

### 5. CNN architecture

The model uses a sequential CNN with three convolution/pooling blocks followed by a dense classification head.

```text
Input: 64 x 64 x 3
        |
        v
Conv2D: 32 filters, 3 x 3, ReLU
        |
MaxPooling2D: 2 x 2
        |
Dropout: 0.25
        |
Conv2D: 64 filters, 3 x 3, ReLU
        |
MaxPooling2D: 2 x 2
        |
Dropout: 0.25
        |
Conv2D: 128 filters, 3 x 3, ReLU
        |
MaxPooling2D: 2 x 2
        |
Flatten
        |
Dense: 128, ReLU
        |
Dropout: 0.50
        |
Dense: 29, Softmax
```

The model contains **686,941 trainable parameters**.

The final `Dense(29, activation='softmax')` layer produces one probability for each of the 29 ASL classes.

### 6. Training configuration

The notebook uses the following main training settings:

| Setting | Value |
|---|---|
| Image size | `(64, 64)` |
| Batch size | `32` |
| Maximum epochs | `10` |
| Optimizer | Adam |
| Loss | Sparse categorical crossentropy |
| Metric | Accuracy |
| Early stopping | Enabled |
| Early stopping patience | 5 epochs |

The model is trained using `ImageDataGenerator` for on-the-fly augmentation.

### 7. Data augmentation

The training pipeline applies random transformations to increase visual variation and reduce the chance of simply memorizing the original training images.

The configured augmentations include:

- Rotation up to 20 degrees
- Width shifting up to 20%
- Height shifting up to 20%
- Shearing up to 20%
- Zooming up to 20%
- Nearest-pixel filling for newly exposed areas

**Horizontal flipping is intentionally disabled.** The notebook notes that horizontally flipping an ASL hand sign may change its meaning, so a mirrored image is not treated as automatically equivalent to the original sign.

### 8. Accuracy and training results

The recorded training run produced the following results:

| Metric | Start | Final |
|---|---:|---:|
| Training accuracy | 13.0% | **74.8%** |
| Validation accuracy | 36.3% | **93.1%** |
| Training loss | 3.00 | 0.75 |
| Validation loss | 2.09 | 0.21 |

The final validation evaluation reported approximately **0.93**, corresponding to about **93% / 93.1% validation accuracy**.

### Why is validation accuracy higher than training accuracy?

The notebook explicitly explains this behavior.

The training images are augmented during training, meaning the model sees rotated, shifted, sheared, and zoomed versions of the images. These transformed examples can be more difficult to classify than the original images.

The validation set, however, is evaluated **without those training-time augmentations**. As a result, it is possible for validation accuracy to be higher than training accuracy in this particular setup.

The notebook also reports that validation loss continued to decrease through the 10-epoch run, so there was no obvious sign of the typical validation-loss increase associated with overfitting during this run.

### 9. Model export

After training, the notebook saves the model in HDF5 format:

```text
asl_classifier_model.h5
```

The model is stored in Google Drive so it persists after the Colab session ends.

The notebook also notes that `.h5` is a legacy Keras format and that newer projects may prefer the native `.keras` format.

### 10. Sanity checks

The notebook performs individual predictions on selected image indices (`0`, `11`, and `9999`) and displays the predicted class beside the actual class.

This is a basic sanity check to confirm that the trained model can produce class predictions on individual examples.

---

## Live Prediction Application — `app.py`

`app.py` is the deployment/inference component of the project. It does not train the CNN again. Instead, it loads the already-trained model and connects it to a webcam-based web application.

### Main technologies used

- **Python** – application language
- **TensorFlow / Keras** – loads and runs the trained CNN
- **OpenCV (`cv2`)** – webcam capture, image conversion, resizing, drawing, and JPEG encoding
- **MediaPipe** – detects and tracks the hand in the webcam feed
- **NumPy** – array and numerical processing
- **Flask** – serves the web page and live video stream

### Model loading

The application loads the trained model with:

```python
keras_model = load_model('asl_classifier_model.h5')
```

Therefore, `asl_classifier_model.h5` must be available in the location expected by the application (the provided code expects it in the same folder as `app.py`).

### MediaPipe hand detection

MediaPipe is used to locate the hand rather than to classify the ASL sign itself.

The application configures MediaPipe Hands to track up to one hand and uses tracking between frames to improve video performance.

Once a hand is detected, the application calculates a bounding rectangle around the hand landmarks and adds a **20-pixel margin** before cropping.

### Image preprocessing during inference

The live webcam crop must match the training input format. `classify_crop()` therefore:

1. Receives the crop in **RGB** format.
2. Resizes it to **64 × 64**.
3. Adds a batch dimension.
4. Converts it to `float32`.
5. Divides pixel values by `255.0`.
6. Passes the result to the Keras model.
7. Selects the highest-probability class with `argmax()`.

This is important because the model was trained on normalized RGB images of the same input size.

### Class prediction

The model outputs probabilities for the 29 classes. `app.py` selects the class with the highest probability and displays it with its confidence value.

For example, the application may display a label in this general form:

```text
A (0.95)
```

where `A` is the predicted class and `0.95` represents the highest predicted probability for that frame.

### Webcam processing flow

For each frame:

```text
OpenCV webcam frame
        |
        v
Convert BGR -> RGB
        |
        v
MediaPipe hand detection
        |
        v
Calculate hand bounding box
        |
        v
Add 20 px margin
        |
        v
Crop hand in RGB
        |
        v
64 x 64 + /255 normalization
        |
        v
Keras CNN prediction
        |
        v
Class + confidence
        |
        v
Draw rectangle + label
        |
        v
Encode frame as JPEG
        |
        v
Stream to Flask page
```

## Flask Web Routes

The application exposes two routes:

### `/`

Loads the Flask template:

```python
return render_template('index.html')
```

This is the main web page.

### `/video_feed`

Streams the processed webcam frames using a multipart response:

```python
mimetype='multipart/x-mixed-replace; boundary=frame'
```

This allows the browser page to display the continuously updated webcam feed produced by `gen_frames()`.

## Important Project Files

| File | Purpose |
|---|---|
| `Solis_WEEK_7_8.ipynb` | Training notebook. Prepares the dataset, trains/evaluates the CNN, and saves the model. |
| `app.py` | Flask/OpenCV/MediaPipe inference application that loads the `.h5` model and performs live ASL prediction. |
| `asl_classifier_model.h5` | Trained Keras CNN used for inference. Generated by the notebook. |
| `templates/index.html` | Flask page rendered by `/` and used for the web interface. |

> The exact template/static files included in a repository should match the Flask application structure used by `app.py`. In particular, `render_template('index.html')` expects the HTML template to be available to Flask.

## Prerequisites

### For model training

The notebook is designed for a Google Colab-style environment and uses:

- Google Colab or another Jupyter-compatible environment
- Google Drive access
- Python with TensorFlow/Keras support
- The ASL Alphabet dataset
- Kaggle dataset access for the notebook's `kagglehub` download path
- Libraries used by the notebook, including:
  - TensorFlow
  - OpenCV
  - NumPy
  - scikit-learn
  - Matplotlib
  - MediaPipe
  - `tqdm`
  - `kagglehub`

The notebook installs MediaPipe explicitly and downloads the dataset through `kagglehub` when the expected dataset directory is not already present in Google Drive.

### For the live Flask application

You need:

- Python environment compatible with the installed TensorFlow/Keras stack
- A working webcam
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Flask
- The trained `asl_classifier_model.h5`
- The Flask HTML template (`templates/index.html`)

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/enfaith1/Solis_Week7-8.git
cd Solis_Week7-8
```

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the required packages

Install the Python libraries required by `app.py` and the training workflow:

```bash
pip install tensorflow opencv-python mediapipe numpy flask scikit-learn matplotlib tqdm kagglehub
```

> TensorFlow installation can depend on the operating system, Python version, CPU/GPU configuration, and the TensorFlow release being used. Use a TensorFlow-supported Python environment when installing the project dependencies.

### 4. Place the trained model in the project

Copy the generated model:

```text
asl_classifier_model.h5
```

into the location expected by `app.py`.

The provided code loads it with:

```python
load_model('asl_classifier_model.h5')
```

### 5. Ensure the Flask template exists

The application renders:

```text
templates/index.html
```

because the root route calls `render_template('index.html')`.

A typical project structure is therefore:

```text
Solis_Week7-8/
├── app.py
├── asl_classifier_model.h5
├── Solis_WEEK_7_8.ipynb
└── templates/
    └── index.html
```

### 6. Run the application

```bash
python app.py
```

The Flask application is configured to run on:

```text
(http://127.0.0.1:5000)
```

Open the Flask address in a web browser, allow webcam access when prompted, and position one hand within the camera view.

## Training the Model Again

The normal training workflow is:

1. Open `Solis_WEEK_7_8.ipynb` in Google Colab.
2. Mount Google Drive.
3. Make the ASL dataset available through the notebook's configured Drive path or allow the notebook to download and copy it.
4. Run the dataset-copy and loading cells.
5. Train the CNN.
6. Review the training and validation metrics.
7. Run the sanity-check predictions.
8. Save the trained model as `asl_classifier_model.h5`.
9. Download/copy the `.h5` file into the local application project.
10. Run `app.py` for webcam inference.

## Important Compatibility Requirements

The training and inference pipelines must remain consistent.

### Input size must remain 64 × 64

The notebook trains with:

```python
IMAGE_SIZE = (64, 64)
```

and `app.py` uses:

```python
KERAS_INPUT_SHAPE = (64, 64)
```

Changing the model's expected input size without retraining/updating the inference pipeline can cause incorrect inputs or runtime errors.

### RGB ordering must be preserved

The notebook reads the training images as RGB. The live application therefore converts OpenCV's BGR webcam frames to RGB before processing the crop.

This prevents a mismatch between the color representation used during training and the color representation used during inference.

### Pixel normalization must remain consistent

The notebook normalizes input using:

```python
image_data = tf.convert_to_tensor(image_data) / 255.0
```

The application uses:

```python
image_array = ... / 255.0
```

Both sides therefore expect pixel values scaled to approximately the `[0, 1]` range.

### Class ordering must remain identical

The notebook creates class names using the sorted folder names. `app.py` manually reproduces that order in `KERAS_CLASSES`.

If the dataset folders or class ordering are changed, the class list in `app.py` must also be updated to match the model's output indices.

## Accuracy Interpretation

The project achieved a **93.1% validation accuracy** during the recorded training run. This is a strong result for the specific validation set and training configuration used in the notebook.

However, validation accuracy is not the same thing as production/webcam accuracy. Several differences can affect real-world performance, including:

- Different lighting conditions
- Different backgrounds
- Hand position and distance from the camera
- Camera quality
- The exact framing produced by the MediaPipe hand crop
- Variations in hand shape, orientation, and signing style
- Differences between the training images and live webcam images

Therefore, the most accurate description is:

> **The CNN achieved 93.1% validation accuracy on the notebook's held-out validation set of 5,800 images.**

## Known Limitations

### 1. One-hand detection

The provided MediaPipe configuration sets:

```python
max_num_hands=1
```

so the live application is configured around a single detected hand.

### 2. Webcam inference is not the same dataset as validation

The model was trained using the ASL image dataset, while live inference uses MediaPipe-generated hand crops from a webcam. The two input distributions are not guaranteed to be identical.

### 3. Confidence is a model probability, not a guaranteed correctness score

The value displayed beside the predicted label comes from the highest softmax output. A high value indicates model confidence in the selected class, but it does not guarantee that the prediction is correct.

### 4. The project recognizes individual ASL classes, not complete continuous sentences

The current application predicts one of the 29 trained categories for each detected frame. It does not implement a complete language model, word segmentation system, or continuous sentence-level ASL translation pipeline.

### 5. `del`, `nothing`, and `space` are special dataset classes

These categories are part of the 29-class training set and are treated as ordinary model outputs. They do not by themselves create a full text-entry or sentence-generation system.

## Technologies

| Technology | Role |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | CNN training and model inference |
| OpenCV | Image processing and webcam capture |
| MediaPipe Hands | Hand detection/tracking and landmarks |
| NumPy | Numerical and array operations |
| scikit-learn | Train/validation split |
| Matplotlib | Dataset visualization and notebook output |
| Flask | Web application and webcam stream |
| Google Colab / Google Drive | Training environment and persistent storage |
| Kaggle / `kagglehub` | ASL dataset retrieval |

## Conclusion

This project demonstrates a complete workflow for image-based ASL recognition:

- preparing and limiting a large image dataset,
- preprocessing and normalizing images,
- training a CNN classifier,
- evaluating the model on a held-out validation set,
- saving the trained Keras model,
- detecting a live hand with MediaPipe,
- preprocessing the webcam crop consistently with the training pipeline,
- and deploying the classifier through a Flask web application.

The recorded model achieved **93.1% validation accuracy** after 10 training epochs on the selected **29,000-image dataset**, while the live application provides real-time predicted ASL classes and confidence values from a webcam.
