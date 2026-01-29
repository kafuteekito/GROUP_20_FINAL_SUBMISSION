# Home Security System
## Real-Time Face Detection and Recognition Application

### 1. Project Description

This project implements a **real-time home security system** based on face detection and face recognition technologies.
The system continuously captures video from a webcam, detects faces in each frame, recognizes known individuals, and triggers alerts when an **unknown person** is detected for a predefined duration.
The application is implemented as a **desktop GUI application using Tkinter** and integrates modern deep-learning–based computer vision models.

### 2. System Architecture

The system follows a modular architecture and consists of the following logical components:

1. **Video Acquisition**
2. **Face Detection**
3. **Face Recognition**
4. **Decision Logic (Known / Unknown)**
5. **Alert System**
6. **Graphical User Interface (GUI)**

Each component is implemented in a separate module and integrated in the main application loop.

### 3. Technologies and Libraries

* **Python 3**
* **OpenCV** – webcam access and image processing
* **RetinaFace** – deep-learning–based face detection
* **ArcFace (DeepFace)** – face recognition and embedding generation
* **NumPy** – numerical operations and vector comparison
* **Tkinter** – desktop graphical user interface
* **PIL (Pillow)** – image conversion for GUI display
* **Threading & Queue** – parallel processing and thread-safe frame exchange
* **Email & Audio utilities** – alert notifications

### 4. Face Detection Module

Face detection is performed using **RetinaFace**, which provides high accuracy and robustness under different lighting and pose conditions.
The function:

```python
detect_faces_retinaface(frame)
```
returns bounding boxes in the form:

```
(x1, y1, x2, y2, confidence)
```

To ensure safe cropping, bounding boxes are clamped to image boundaries using:

```python
clamp_box(x1, y1, x2, y2, width, height)
```

This prevents index errors and invalid image slices.

### 5. Face Recognition Module (ArcFace)

Face recognition is based on **ArcFace embeddings**, which are generated using the DeepFace framework.

Workflow:

1. A detected face region is cropped from the frame
2. An embedding vector is computed using ArcFace
3. The embedding is compared against stored embeddings using **cosine distance**

```python
embed_face_arcface(face_image)
identify(embedding, gallery, threshold)
```

The recognition result is:

* a **person’s name** if the distance is below the threshold
* **"Unknown"** otherwise

The recognition threshold is loaded dynamically from `gallery_db.json`.

### 6. Face Database

Known faces are stored in a local JSON file:

```
gallery_db.json
```

Structure:

```json
{
  "threshold": 0.40,
  "people": {
    "PersonName": [
      [embedding_1],
      [embedding_2]
    ]
  }
}
```

This allows:

* flexible threshold tuning
* multiple embeddings per person
* easy extension without retraining models

### 7. Decision Logic

The system distinguishes between three states:

#### 7.1 Known Face

* At least one recognized person is present
* After `KNOWN_WELCOME_SECONDS`, a **welcome message** is shown:

  ```
  Welcome, <Name>!
  ```
* Unknown timers are reset

#### 7.2 Unknown Face

* Only unknown faces are present
* A timer starts
* Status shows elapsed detection time

#### 7.3 Alert Trigger

If an unknown face remains present for:

```
UNKNOWN_ALERT_SECONDS
```

the system:

1. Saves a screenshot to `unknown_captures/`
2. Plays an alarm sound
3. Sends an email notification with the image attached


### 8. Alert System

The alert system is implemented asynchronously to avoid blocking the main video loop.

Alerts include:

* **Audio alarm** (`Alarm.wav`)
* **Email notification**
* **Image evidence**


### 9. Graphical User Interface (GUI)

The GUI is built using **Tkinter** and follows a modern dark theme.

Displayed elements:

* Live video stream
* Bounding boxes and recognition labels
* System status indicator (colored dot)
* FPS counter
* Timestamp


### 10. Performance Considerations

To ensure real-time performance:

* Frames are resized to `FRAME_WIDTH`
* Recognition is executed every `PROCESS_EVERY_N_FRAMES`
* Detection and recognition run in a background thread
* GUI rendering is decoupled from video processing

FPS is calculated and displayed in real time.


### 11. Error Handling and Stability

The system includes:

* Safe thread termination
* Camera release on exit
* Protection against empty frames
* Graceful handling of missing database or email credentials


### 12. Use Cases

* Home entrance monitoring
* Office access supervision
* Educational demonstration of AI-based security systems
* Prototype for further IoT or smart-home integration


### 13. Conclusion

This project demonstrates a **complete end-to-end AI application**, combining computer vision, deep learning, real-time processing, and GUI design.

It showcases:

* Practical application of face detection and recognition
* Modular software architecture
* Real-time constraints handling
* Integration of AI with user-facing systems

The system can be easily extended with:

* GPU acceleration
* Additional detectors
* Cloud-based databases
* Mobile or web interfaces
