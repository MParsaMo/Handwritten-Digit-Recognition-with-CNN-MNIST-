# Handwritten-Digit-Recognition-with-CNN-MNIST-
This project uses a **Convolutional Neural Network (CNN)** to classify handwritten digits from the famous **MNIST dataset**. The model is built using **Keras** and **TensorFlow**.
---

## 🚀 Project Overview

- 📊 Dataset: MNIST (70,000 grayscale images of handwritten digits, 28x28 pixels)
- 🔍 Task: Multi-class classification (10 digits: 0-9)
- 🏗 Model: Convolutional Neural Network (CNN) with Batch Normalization and Dropout

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

---

📝 How It Works
1️⃣ Data Loading & Preprocessing

Load MNIST dataset directly via Keras.

Split original training data into:

Training set (80%)

Validation set (20%)

Reshape images for CNN: (28, 28, 1)

Normalize pixel values to range [0, 1]

One-hot encode labels.

2️⃣ Model Architecture
| Layer Type             | Details                             |
| ---------------------- | ----------------------------------- |
| Conv2D                 | 32 filters, (3x3), ReLU, BatchNorm  |
| Conv2D                 | 32 filters, (3x3), ReLU, BatchNorm  |
| MaxPooling2D + Dropout | (2x2), Dropout 25%                  |
| Conv2D                 | 64 filters, (3x3), ReLU, BatchNorm  |
| Conv2D                 | 64 filters, (3x3), ReLU, BatchNorm  |
| MaxPooling2D + Dropout | (2x2), Dropout 25%                  |
| Flatten                |                                     |
| Dense                  | 512 units, ReLU, BatchNorm, Dropout |
| Output Dense           | 10 units (Softmax)                  |

3️⃣ Compilation

Loss: categorical_crossentropy

Optimizer: adam

Metrics: accuracy

4️⃣ Training

Trained for 10 epochs with batch size 128.

Evaluated on unseen test set.

---

📊 Sample Output
x_train.shape: (48000, 28, 28)
x_val.shape: (12000, 28, 28)
x_test.shape: (10000, 28, 28)

Final Test Loss: 0.0345
Final Test Accuracy: 0.9895

---

🧩 Key Features
✅ Modern CNN with Batch Normalization and Dropout
✅ Training/Validation/Test split for robust evaluation
✅ High accuracy on handwritten digit recognition

---

💡 Next Steps / Improvements
Increase number of epochs for higher accuracy.

Experiment with more complex CNNs (e.g., ResNet, LeNet).

Try different optimizers or learning rates.

Apply to different datasets (Fashion-MNIST, EMNIS)

---

🙋 Contributing
Feel free to fork, improve the model, or adapt it to new datasets!

---

👉 Let me know if you want:
- Jupyter Notebook version
- Model saving and loading (`.h5` file)
- Real-time digit prediction demo

