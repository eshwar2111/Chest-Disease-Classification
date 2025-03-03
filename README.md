# Chest Disease Classification (Pneumonia Detection)

##  Project Overview
This project implements a **multimodal deep learning model** for **chest disease classification**, specifically detecting **pneumonia** from **chest X-ray images** and corresponding **text reports**. The model combines **CNN for image processing** and **BiLSTM with Attention for text analysis**.

##  Dataset
The dataset is structured as follows:
```
/kaggle/input/chest-xray-pneumonia/
    ├── chest_xray/
        ├── train/
            ├── NORMAL/
            ├── PNEUMONIA/
        ├── val/
            ├── NORMAL/
            ├── PNEUMONIA/
        ├── test/
            ├── NORMAL/
            ├── PNEUMONIA/
```
- **NORMAL**: Healthy chest X-rays.
- **PNEUMONIA**: X-rays showing pneumonia.

##  Dependencies
- Python 3.x
- TensorFlow/Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

To install dependencies:
```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

## Model Architecture
- **Image Model (CNN)**: Extracts features from X-ray images using convolutional layers.
- **Text Model (BiLSTM + Attention)**: Processes medical text reports using bidirectional LSTM and attention mechanism.
- **Fusion Model**: Combines the extracted features from both modalities and classifies the case as "Normal" or "Pneumonia".

##  Training the Model
Run the following command to train the model:
```python
history = model.fit(
    [X_train_img, X_train_txt], y_train,
    validation_data=([X_val_img, X_val_txt], y_val),
    batch_size=32,
    epochs=10
)
```

##  Model Evaluation
After training, evaluate the model using:
```python
test_loss, test_acc = model.evaluate([X_test_img, X_test_txt], y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

##  Making Predictions
To predict the diagnosis of a new case:
```python
def predict_diagnosis(image_path, text_report):
    img = load_img(image_path, target_size=(224, 224), color_mode="grayscale")
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    text_seq = pad_sequences(tokenizer.texts_to_sequences([text_report]), maxlen=50)
    prediction = model.predict([img, text_seq])[0][0]
    return "Pneumonia" if prediction > 0.5 else "Normal"
```

##  Issues & Potential Fixes
- **Overfitting Detected**: The model reaches 100% accuracy too quickly.
    - 🔹 Add **Dropout layers**
    - 🔹 Use **data augmentation**
    - 🔹 Reduce model complexity
- **Loss Values Too Small**: Use `categorical_crossentropy` and reduce learning rate.
- **Imbalanced Dataset**: Apply class weights or oversampling.
- **Attention Layer Incorrectly Used**: Modify self-attention mechanism.

##  License
This project is licensed under the MIT License.

---
**Author:** Eshwar B. 

