# Lab1 Project Report

## 1. Introduction
This report presents the implementation and comparison of three classification models: Logistic Regression, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP). The goal is to evaluate their performance in terms of accuracy and training time.

## 2. Methodology

### 2.1. Data Preprocessing
The dataset was preprocessed using feature scaling with `StandardScaler` to ensure all features are on the same scale.

### 2.2. Model Implementation
- **Logistic Regression**: Implemented using gradient descent to minimize the logistic loss.
- **SVM**: Implemented using hinge loss and gradient descent.
- **MLP**: Implemented using PyTorch with one hidden layer and ReLU activation.

### 2.3. Parameter Settings
- **Logistic Regression**: Learning rate = 0.1, Max iterations = 5000
- **SVM**: Learning rate = 0.001, Lambda = 0.01, Max iterations = 1000
- **MLP**: Hidden size = 64, Learning rate = 0.001, Epochs = 100

## 3. Results

### 3.1. Logistic Regression
- **Training Loss**: Decreased steadily over epochs.
- **Accuracy**: Both training and test accuracy converged to high values.

### 3.2. SVM
- **Training Loss**: Decreased steadily over epochs.
- **Accuracy**: Both training and test accuracy improved over epochs.

### 3.3. MLP
- **Training Loss**: Decreased steadily over epochs.
- **Accuracy**: Both training and test accuracy improved over epochs.

### 3.4. Model Comparison
- **Train Accuracy**: Logistic Regression, SVM, MLP
- **Test Accuracy**: Logistic Regression, SVM, MLP
- **Training Time**: Logistic Regression, SVM, MLP

## 4. Conclusion
The MLP model achieved the highest test accuracy and required the least training time. The Logistic Regression model had the most training time but slightly lower accuracy. The SVM model offered a balance between accuracy and training time. Based on the results, the choice of model depends on the specific requirements of the application.

## 5. Visualizations

### 5.1. Loss and Accuracy Plots
![all_models_comparison](https://github.com/user-attachments/assets/f390bcdc-610d-4650-a5cd-d219ee03818f)


### 5.2. Model Comparison
![model_comparison](https://github.com/user-attachments/assets/971bc5ee-78f0-4253-8fbc-49da1df2fea3)


## 6. Future Work
- Experiment with different hyperparameters to further optimize model performance.
- Explore other advanced architectures for potentially better results.

## 7. References
- Project code and data available in the provided repository.
- Large language models like deepseek and kimi.
