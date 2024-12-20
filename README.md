# 🌟 **Animal Classification Project** 🌟

## 📜 **Project Overview**
This project demonstrates the application of a Convolutional Neural Network (CNN) to classify animal images from the **Animals with Attributes 2** dataset. The pipeline involves image preprocessing, model training, testing on various datasets, and performance evaluation under different conditions.

---

## 🚀 **Key Features**
- **Dataset Preprocessing:** Resize and normalize images.
- **Data Augmentation:** Brightness/contrast adjustments and white balancing.
- **Model Training:** Fine-tuned CNN for multi-class classification.
- **Evaluation:** Tested on normal, manipulated, and white-balanced datasets.

---

## 🛠️ **Technologies Used**
- **Programming Language:** Python 🖊️
- **Libraries:**
  - TensorFlow & Keras for deep learning.
  - OpenCV for image manipulation.
  - NumPy for numerical computations.
  - Matplotlib for visualization.
- **Dataset:** [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/)

---
## 🔄 **Workflow**

### 📊 **1. Dataset Preprocessing**
- **Resizing:** All images were resized to **128x128 pixels** to standardize input dimensions, ensuring that the model can handle a consistent image size during training and testing.
- **Normalization:** Pixel values were normalized to the range `[0, 1]` to help the neural network process the image data efficiently. This step prevents large pixel values from dominating the training process and allows the optimizer to work more effectively.

### 🎨 **2. Data Manipulation**
- **Brightness and Contrast Adjustment:** OpenCV was used to adjust the brightness and contrast of the images. This technique is aimed at improving image quality and helping the model become more robust to changes in lighting conditions.
- **White Balancing:** The Gray World Algorithm was applied to correct the color balance of images. This step is useful for ensuring that the model can perform consistently across images taken in different lighting conditions.

### 🧠 **3. Model Training**
- **Architecture:** A Convolutional Neural Network (CNN) was chosen due to its effectiveness in image classification tasks. CNNs are designed to automatically detect features like edges, textures, and patterns from images.
- **Optimizer:** Adam optimizer was selected for its adaptive learning rate, which can help the model converge faster and avoid overshooting during training.
- **Loss Function:** Categorical Crossentropy was used because the task involves multi-class classification. This loss function is ideal for tasks where each sample belongs to one of several classes.
- **Metrics:** Accuracy was used as the primary metric to evaluate the model's performance during training and testing. It measures how often the model's predictions match the actual labels.
- **Training Parameters:**
  - **Epochs:** The model was trained for **10 epochs**. This number could be increased if the model hasn't fully converged, but more epochs also risk overfitting.
  - **Batch Size:** A batch size of **32** was used, which is a common choice for image classification tasks. This size helps balance memory efficiency and training stability.

### 📉 **4. Evaluation**
- **Training Test Accuracy:** The model achieved an **accuracy of 83%** on the training test, indicating good learning performance during training.
- **Normal Dataset:** The performance of the model on the untouched test set (normal dataset) showed an accuracy of **68.41%**, with a loss of **1.03**.
- **Manipulated Dataset:** The model's performance drastically dropped on the manipulated dataset (with adjusted brightness and contrast), achieving only **10.05% accuracy** and a significantly higher loss of **188.63**. This suggests that the model struggled to generalize to these manipulated images.
- **White-Balanced Dataset:** Similarly, the white-balanced dataset produced poor results with **38.46% accuracy** and a loss of **569.64**, indicating that the color correction applied might have introduced artifacts that confused the model.

---

### **Performance Summary Table:**

| **Dataset Type**      | **Accuracy (%)** | **Loss**     |
|-----------------------|------------------|--------------|
| **Normal**            | 68.41            | 1.03         |
| **Manipulated**       | 10.05            | 188.63       |
| **White-Balanced**    | 38.46            | 569.64       |

---

### 🧠 **Discussion & Future Improvements**

#### **Challenges:**
- The manipulated and white-balanced datasets performed poorly, which could suggest a few things:
  - The model may not have learned the right features during training, making it unable to generalize well to transformed images.
  - The transformations (brightness/contrast adjustment and white balancing) may have altered the image features in ways that the model couldn't handle.

#### **Proposed Solutions:**
1. **Data Augmentation:**
   - To make the model more robust, augmenting the training dataset with manipulated and white-balanced images would help it learn to handle a wider variety of image transformations.
   
2. **Transfer Learning:**
   - Experimenting with transfer learning using pre-trained models like **ResNet** or **EfficientNet** could significantly improve performance. These models are already trained on large and diverse datasets (e.g., ImageNet) and can be fine-tuned to your specific task, allowing the model to leverage knowledge learned from different image domains.
   
3. **Model Optimization:**
   - Improving the model architecture could help, such as:
     - Adding more convolutional layers to capture more intricate patterns in the images.
     - Implementing dropout to prevent overfitting by randomly ignoring certain neurons during training, helping the model generalize better to unseen data.
   
4. **Hyperparameter Tuning:**
   - Fine-tuning the number of epochs, batch size, and learning rate could further optimize the training process. For example, increasing the number of epochs may help if the model hasn't converged fully within 10 epochs.

5. **Regularization Techniques:**
   - Applying regularization methods like **L2 regularization** could help prevent overfitting, particularly when working with small or imbalanced datasets.

By incorporating these strategies, the model can be better prepared to handle manipulated and white-balanced images, leading to improved accuracy and reduced loss.

---

