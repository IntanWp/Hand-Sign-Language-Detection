# ASL Hand Sign Language Recognition

This project is a web-based American Sign Language (ASL) hand sign recognition system using computer vision and machine learning. It uses [Flask](https://flask.palletsprojects.com/), [OpenCV](https://opencv.org/), and [MediaPipe](https://mediapipe.dev/) for real-time hand tracking and a Random Forest classifier for gesture recognition.

## Features

- Real-time hand sign recognition via webcam
- Feature extraction from hand landmarks using MediaPipe
- Model training pipeline for custom datasets
- Web interface for live prediction

## Project Structure

- [`app.py`](app.py): Flask web server for real-time prediction and webcam streaming.
- [`new_dataset_model.py`](new_dataset_model.py): Script for dataset processing, feature extraction, model training, and evaluation.
- `requirements.txt`: Python dependencies.

## Setup

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**

   - Organize your ASL images in folders (one folder per class) as described in the code comments.
   - Example:
     ```
     asl_alphabet_train/
       A/
         img1.jpg
         img2.jpg
         ...
       B/
         img1.jpg
         ...
     ```

3. **Process dataset and train the model:**

   ```sh
   python new_dataset_model.py --dataset asl_alphabet_train --tune
   ```
   - Use `--tune` for hyperparameter tuning (optional).
   - The trained model will be saved as `rf_model.p`.

4. **Run the web app:**

   ```sh
   python app.py
   ```
   - Open your browser at [http://localhost:5000](http://localhost:5000).

## Usage

- The web app will show a live webcam feed.
- Detected hand signs will be displayed with their predicted label and confidence.

## Notes

- Make sure your webcam is connected and accessible.
- The model expects the same feature extraction as in training (`extract_enhanced_features`).
- You can retrain the model with your own dataset using [`new_dataset_model.py`](new_dataset_model.py).

## Requirements

See [`requirements.txt`](requirements.txt) for the full list.

---

**Credits:**  
- [MediaPipe](https://mediapipe.dev/) for hand tracking  
- [OpenCV](https://opencv.org/) for image processing  
- [Flask](https://flask.palletsprojects.com/) for the
