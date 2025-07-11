# 🖐️ Hand Gesture Recognition using CNN

This project is a part of my internship task where I built a hand gesture recognition model using Convolutional Neural Networks (CNN). The model classifies different hand gestures from grayscale images.

---

## 📁 Dataset

- **Name:** LeapGestRecog  
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)  
- **Details:** The dataset contains grayscale images of 10 different hand gestures collected using a Leap Motion Controller. Each gesture is stored in separate folders, and images are pre-labeled accordingly.

---

## 📁 Download Trained Model

[Click here to download hand_gesture_model.h5](https://drive.google.com/drive/folders/14bPpTP3zFhXcWEAQS1ojpjKNO21h7xxT?usp=drive_link)

---

## 🔧 Libraries Used

- Python  
- OpenCV  
- NumPy  
- Matplotlib  
- scikit-learn  
- TensorFlow / Keras  

---

## 📌 Project Steps

### 1. Dataset Download
Dataset was downloaded using `opendatasets` directly from Kaggle.

### 2. Data Preprocessing
- Converted images to grayscale  
- Resized all images to `64x64`  
- Normalized pixel values (0–1)  
- One-hot encoded gesture labels  
- Final shape: `(num_samples, 64, 64, 1)`

### 3. Train-Test Split
Split the dataset using `train_test_split` with an 80-20 ratio and stratified labels.


### 4. Model Building (CNN)
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])









5. Model Training
Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 10

Batch Size: 64

6. Evaluation
Achieved high training and validation accuracy

Plotted accuracy/loss graphs

Evaluated with classification_report and confusion_matrix

📊 Results
Model successfully classifies all 10 gesture classes

Achieved good validation accuracy

Can be used for gesture-based Human-Computer Interaction systems

📷 Sample Gestures
Palm | Fist | L | Thumb | Index | Others
(Images are from the dataset)

🚀 Future Improvements
Increase epochs and batch size for better accuracy

Implement data augmentation

Try real-time prediction with webcam (if needed)

📌 Note
This project does not include real-time webcam input — it only classifies gestures from image dataset as per internship task requirements.

🙋‍♂️ Author
Md Shahfahad Khan
