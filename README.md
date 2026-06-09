# Handwritten Digit Recognition Using TensorFlow 🧠🔢

A Deep Learning project that recognizes handwritten digits (0–9) using the MNIST dataset and a Neural Network built with TensorFlow and Keras.

## Description

This project trains a neural network to identify handwritten digits from images. The model learns patterns from thousands of examples in the MNIST dataset and predicts the digit shown in a new image.

This project is an excellent introduction to:

- Deep Learning
- Neural Networks
- Computer Vision
- TensorFlow & Keras
- Image Classification

## Dataset

The project uses the **MNIST Dataset**, one of the most famous datasets in Machine Learning.

### Dataset Information

- 70,000 handwritten digit images
- Digits from 0 to 9
- Image size: 28 × 28 pixels
- Grayscale images

| Dataset | Images |
|----------|---------:|
| Training Set | 60,000 |
| Test Set | 10,000 |

## Requirements

Install the required libraries:

```python
!pip install tensorflow matplotlib
```

Or from the terminal:

```bash
pip install tensorflow matplotlib
```

## Code

```python
# 1. Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 2. Load Dataset (MNIST)
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 3. Preprocess (Normalize pixel values between 0 and 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# 4. Build Neural Network
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 5. Compile and Train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training started...")
model.fit(X_train, y_train, epochs=5)

# 6. Assessment Results
test_loss, test_acc = model.evaluate(
    X_test,
    y_test,
    verbose=2
)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# 7. Visualize a prediction
prediction = model.predict(X_test[:1])

print(
    f"Predicted Digit: {prediction.argmax()}"
)

plt.imshow(X_test[0], cmap='gray')
plt.show()
```

## How to Run

1. Install Python 3.
2. Install required libraries:

```bash
pip install tensorflow matplotlib
```

3. Save the code as:

```text
digit_recognition.py
```

4. Run:

```bash
python digit_recognition.py
```

## Example Output

```text
Training started...

Epoch 1/5
...

Test Accuracy: 97.80%

Predicted Digit: 7
```

A handwritten digit image will also be displayed.

## Features

- Uses the famous MNIST dataset
- Deep Learning with TensorFlow
- Handwritten digit recognition
- Model evaluation with accuracy score
- Visual prediction display
- Beginner-friendly AI project

## Concepts Used

### Data Loading

```python
mnist.load_data()
```

Loads the MNIST handwritten digit dataset.

### Data Normalization

```python
X_train / 255.0
```

Converts pixel values from:

```text
0–255
```

to:

```text
0–1
```

which improves training performance.

### Flatten Layer

```python
layers.Flatten()
```

Converts a 28×28 image into a single vector.

### Dense Layer

```python
layers.Dense()
```

Fully connected neural network layer.

### ReLU Activation

```python
activation='relu'
```

Introduces non-linearity into the network.

### Dropout Layer

```python
layers.Dropout(0.2)
```

Helps prevent overfitting during training.

### Softmax Output Layer

```python
layers.Dense(10, activation='softmax')
```

Produces probabilities for digits 0–9.

## Deep Learning Workflow

```text
MNIST Images
      ↓
Preprocessing
      ↓
Neural Network
      ↓
Training
      ↓
Testing
      ↓
Digit Prediction
```

## Model Architecture

```text
Input Image (28x28)
          ↓
      Flatten
          ↓
 Dense (128 neurons)
          ↓
      Dropout
          ↓
 Dense (10 neurons)
          ↓
      Softmax
          ↓
 Predicted Digit
```

## Project Structure

```text
handwritten-digit-recognition/
│
├── digit_recognition.py
├── README.md
└── requirements.txt
```

## Skills Demonstrated

- Python Programming
- TensorFlow
- Keras
- Deep Learning
- Computer Vision
- Neural Networks
- Image Classification
- Model Evaluation

## Future Improvements

- Build a true Convolutional Neural Network (CNN)
- Save and load trained models
- Create a web application using Streamlit
- Allow users to upload their own digit images
- Add confusion matrix visualization
- Compare different neural network architectures
- Deploy the model online

## Learning Outcomes

By completing this project, you will learn:

- How neural networks work
- How image classification is performed
- How TensorFlow and Keras are used in real-world AI projects
- How to train, evaluate, and test Deep Learning models

## License

This project is open source and free to use.
