

# **Hand Gesture Recognition System**

This project implements a real-time **Hand Gesture Recognition System** using MediaPipe for hand landmark detection and a custom-trained LSTM model for gesture classification. The system processes video input to detect and classify various hand gestures, making it suitable for applications like sign language recognition and human-computer interaction.

---

## **Table of Contents**
1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Dataset and Model](#dataset-and-model)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## **Features**
- **Real-Time Gesture Recognition**: Detects and classifies hand gestures in live video streams.
- **MediaPipe Integration**: Utilizes MediaPipe for precise hand landmark detection.
- **Custom LSTM Model**: Trained to recognize specific gestures based on a sequence of hand landmarks.
- **Action Labels and Voice Output**: Displays action labels on-screen and provides voice feedback for detected gestures.
- **Error Handling**: Automatically handles cases when no action is detected.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries and Tools**:
  - [OpenCV](https://opencv.org/) - For video processing and webcam input.
  - [MediaPipe](https://mediapipe.dev/) - For hand landmark detection.
  - [TensorFlow](https://www.tensorflow.org/) - For building and deploying the LSTM model.
  - [pyttsx3](https://pypi.org/project/pyttsx3/) - For text-to-speech voice output.
- **Hardware Requirements**:
  - Webcam (for live detection).

---

## **Dataset and Model**
- **Dataset**: The model was trained on hand gesture sequences captured using MediaPipe landmarks.
- **Model**: A Long Short-Term Memory (LSTM) network trained to classify gestures based on temporal sequences of hand keypoints.
- **Output Labels**: The system recognizes actions such as:
  - "Hello"
  - "Thanks"
  - "Yes"
  - "No"
  - "Stop" (add or customize as per your model).

---

## **Installation**
### **Prerequisites**
- Python 3.8 or higher
- pip package manager

### **Steps**
To create your own model in this program and get outputs, follow these steps:

---

### **Step 1: Set Up the Environment**
1. **Install required libraries**:
   Ensure you have the following Python libraries installed:  
   ```
   pip install numpy pandas opencv-python mediapipe tensorflow scikit-learn matplotlib seaborn
   ```

2. **Organize your workspace**:
   - Create a project directory with the following structure:
     ```
     project/
     ├── dataset11/           # For raw datasets (CSV files)
     ├── cleaned_dataset5/    # For cleaned datasets
     ├── models/              # For storing trained models
     └── scripts/             # Python scripts (including this program)
     ```

---

### **Step 2: Data Collection**
1. **Run the data collection script**:
   - Execute the `collect_data()` function in the program.
   - The program will:
     - Start your webcam and detect hand landmarks using MediaPipe.
     - Allow you to perform gestures (e.g., "one", "two", etc.).
     - Save the captured data for each gesture in the `dataset11/` folder as CSV files.
   - You can add more gestures to the `actions` list to capture additional gestures.

2. **Tips during data collection**:
   - Ensure good lighting and a clear background for better hand detection.
   - Perform gestures distinctly, with visible hand movements.
   - Collect at least 50 samples per gesture for a balanced dataset.

---

### **Step 3: Clean the Dataset**
1. **Run the dataset renaming script**:
   - Execute the section of the program labeled *"RENAMING THE DATASET AND SAVE THEM IN NEW FOLDER"*.
   - This script will:
     - Remove unnecessary timestamps from file names.
     - Copy and rename the files into the `cleaned_dataset5/` folder.

2. **Verify the cleaned data**:
   - Open the CSV files in the `cleaned_dataset5/` folder.
   - Ensure that each file contains rows of hand landmark features and the corresponding labels.

---

### **Step 4: Train the Model**
1. **Set up the model training script**:
   - Update the `dataset_dir` variable to point to the `cleaned_dataset5/` folder.
   - Set the `model_name` variable to a descriptive name (e.g., `"sign_language_testing5_multi.h5"`).

2. **Run the training script**:
   - The script will:
     - Load and preprocess the cleaned dataset.
     - Train a dense neural network model using TensorFlow.
     - Save the trained model in the `models/` folder.

3. **Check training results**:
   - View training/validation accuracy and loss graphs.
   - Analyze the confusion matrix to assess the model's performance on each gesture.

---

### **Step 5: Live Detection**
1. **Load the trained model**:
   - Ensure that the trained model file (e.g., `"sign_language_testing5_multi.h5"`) is in the `models/` folder.
   - Use the `load_model()` function to load the model.

2. **Start live detection**:
   - Use the live detection script to:
     - Capture video from the webcam.
     - Detect hand landmarks in real time.
     - Predict gestures using the trained model.
   - Display the detected gesture on the video feed.

---

### **Step 6: Customize the Program**
1. **Add new gestures**:
   - Update the `actions` list in the data collection section with new gestures.
   - Recollect data, clean datasets, and retrain the model.

2. **Improve model performance**:
   - Increase the number of samples per gesture.
   - Experiment with different neural network architectures (e.g., LSTM for temporal data).

3. **Enhance output features**:
   - Add audio announcements for detected gestures using a text-to-speech library like `pyttsx3`.
   - Display bounding boxes and gesture names on the live feed.

---

### **Step 7: Final Outputs**
1. **Save predictions**:
   - Modify the live detection script to log detected gestures in a CSV file for analysis.

2. **Deploy the model**:
   - Integrate the trained model into an application (e.g., a Flask web app, a Flutter mobile app, or a desktop application).

---

**Example Commands to Run the Program**:
1. Collect data:
   ```
   python collect_data.py
   ```
2. Clean datasets:
   ```
   python clean_dataset.py
   ```
3. Train the model:
   ```
   python train_model.py
   ```
4. Run live detection:
   ```
   python live_detection.py
   ```

By following these steps, you can successfully create your own gesture recognition model, train it, and use it for real-time predictions! Let me know if you need any further clarifications.
---

## **Usage**
1. Run the script:
   ```bash
   python gesture_recognition.py
   ```

2. The system will:
   - Open the webcam and display the live video feed.
   - Detect hand landmarks in real-time.
   - Display and announce the detected gesture (if recognized).

3. Press `q` to quit the application.

---

## **Project Structure**
```
Hand-Gesture-Recognition/
│
├── gesture_recognition.py      # Main script for gesture recognition
├── model.h5                    # Trained LSTM model (replace with your own model)
├── requirements.txt            # Dependencies for the project
├── README.md                   # Project documentation
├── utils/                      # Utility functions and classes
│   ├── mediapipe_helper.py     # MediaPipe helper for hand detection
│   ├── preprocessing.py        # Data preprocessing utilities
│   └── audio_feedback.py       # Text-to-speech helper
└── data/                       # Directory for datasets or input files
```

---

## **Results**
The system successfully detects and classifies gestures with high accuracy in real-time. It also provides voice feedback, enhancing accessibility and interactivity.

---

## **Contributing**
Contributions are welcome! If you'd like to improve the project or add new features:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

---
