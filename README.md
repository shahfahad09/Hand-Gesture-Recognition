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

```python
!pip install opendatasets --quiet
import opendatasets as od
od.download("https://www.kaggle.com/datasets/gti-upm/leapgestrecog")
2. Data Preprocessing
Converted images to grayscale

Resized all images to 64x64

Normalized pixel values (0–1)

One-hot encoded gesture labels

Final shape: (num_samples, 64, 64, 1)

3. Train-Test Split
Split the dataset using train_test_split with an 80-20 ratio and stratified labels.

python
Copy
Edit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
4. Model Building (CNN)
python
Copy
Edit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
5. Model Training, Evaluation & Results
📌 Training the Model
python
Copy
Edit
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)
📈 Accuracy & Loss Graphs
python
Copy
Edit
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.legend()

plt.show()
📊 Evaluation Report
python
Copy
Edit
from sklearn.metrics import classification_report
import numpy as np

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))
6. Save the Model
python
Copy
Edit
model.save("hand_gesture_model.h5")
✅ Results
Model successfully classifies all 10 gesture classes

Achieved good training and validation accuracy

Plotted loss/accuracy graphs

Evaluated using classification report

Useful for gesture-based Human-Computer Interaction systems

📷 Sample Gestures
Palm | Fist | L | Thumb | Index | Others
(Images are from the dataset)

🚀 Future Improvements
Increase training epochs and batch size

Apply image augmentation

Try real-time prediction with webcam (OpenCV)

📌 Note
This project does not include real-time webcam input — it only classifies gestures from the image dataset as per internship task requirements.

🙋‍♂️ Author
Md Shahfahad Khan
BCA Student | Machine Learning Intern

yaml
Copy
Edit
