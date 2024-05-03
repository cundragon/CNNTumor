# Brain Tumor Detection

## Introduction
Brain tumors are a significant health concern due to their effects on cognitive function, motor skills, and overall quality of life. Detecting brain tumors can be challenging because of their location in the complex brain structure, similarity to normal tissue, and varying tumor types. This project aims to develop a machine learning model that can classify brain MRI images into different categories, contributing to the early detection of brain tumors and facilitating prompt and effective treatment strategies.

## Dataset
The dataset used in this project consists of brain MRI images categorized into four classes: glioma, meningioma, pituitary, and no tumor. The dataset can be accessed from the following Google Drive link:
- [Brain Tumor Dataset](https://drive.google.com/drive/folders/1Tz46RAAP8Q4K_iBGTlTNQ1EXNMuJcOqn?usp=sharing)

## Methodology
The project utilizes Convolutional Neural Networks (CNNs) to classify brain MRI images. Three different models were developed and trained:

1. **Model 1 (Binary Classification)**: This model is trained to classify brain MRI images into two categories: meningioma and no tumor. It uses a CNN architecture with Conv2D layers, ReLU activation function, and the Adam optimizer. The model achieved high accuracy in predicting the presence or absence of meningioma tumors.

2. **Model 2 (Multi-class Classification)**: This model is an extension of Model 1 and is trained to classify brain MRI images into four categories: glioma, meningioma, pituitary, and no tumor. It uses a similar CNN architecture but replaces the sigmoid activation function with softmax for multi-class classification. The model achieved good accuracy for most classes but struggled with meningioma classification, possibly due to overfitting.

3. **Model 3 (Multi-class Classification with Dropout)**: To address the overfitting issue in Model 2, dropout regularization was added to the CNN architecture. This model showed improved accuracy for meningioma classification while maintaining high accuracy for other classes.

## Results
The performance of each model was evaluated using accuracy metrics on unseen test data. The results are as follows:

1. **Model 1 (Binary Classification)**:
   - Meningioma: 94.44% accuracy
   - No Tumor: 98.52% accuracy

2. **Model 2 (Multi-class Classification)**:
   - Meningioma: 72.88% accuracy
   - No Tumor: 99.26% accuracy
   - Glioma: 93.33% accuracy
   - Pituitary: 100.00% accuracy

3. **Model 3 (Multi-class Classification with Dropout)**:
   - Meningioma: 84.31% accuracy
   - No Tumor: 98.52% accuracy
   - Glioma: 94.00% accuracy
   - Pituitary: 98.67% accuracy

## Conclusion
The project demonstrates the effectiveness of Convolutional Neural Networks in detecting brain tumors from MRI images. The binary classification model (Model 1) achieved high accuracy in distinguishing between meningioma and no tumor cases. The multi-class classification models (Model 2 and Model 3) showed promising results in classifying different types of brain tumors, with Model 3 addressing overfitting issues and improving meningioma classification accuracy.

The project highlights the potential of machine learning in assisting medical professionals in the early detection and management of brain tumors. Further improvements can be made by exploring different CNN architectures, fine-tuning hyperparameters, and incorporating larger and more diverse datasets.

## Usage
To run the models and reproduce the results, follow these steps:
1. Download the brain tumor dataset from the provided Google Drive link and extract it to a suitable location.
2. Install the necessary dependencies and libraries (e.g., TensorFlow, Keras).
3. Run the Jupyter Notebook files for each model:
   - Model 1: `cnn.ipynb`
   - Model 2: `cnn-multi.ipynb`
   - Model 3: `cnn-multi-2.ipynb`
4. Modify the code as needed to point to the correct dataset location and adjust any hyperparameters.
5. Execute the notebook cells to train the models and evaluate their performance.

## License
This project is licensed under the [MIT License](LICENSE).
