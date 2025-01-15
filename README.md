# Disease-Prediction
Classify Disease based on Deep learning
Plant Disease Prediction
This project predicts plant disease status (Bright Blind, Healthy, Late Blight, or Early Blight) using deep learning techniques, specifically TensorFlow.

Overview
The objective of this project is to create a model capable of classifying plant images into one of the following categories:

Healthy: No disease.
Bright Blind: Plants showing brightness abnormalities.
Late Blight: A fungal disease affecting plants.
Early Blight: A bacterial or fungal disease at an early stage.
The model uses TensorFlow, a powerful machine learning framework, to classify images of plant leaves and help with disease detection and early intervention.

Key Features
Data Preprocessing: Data augmentation and normalization techniques to improve model performance.
Model Architecture: A Convolutional Neural Network (CNN) built using TensorFlow to classify plant diseases.
Training: The model is trained on a large dataset of plant images and validated using a separate test dataset.
Real-time Prediction: The trained model can be used for predicting plant health status from new images.
Installation
To use this project, ensure you have the following dependencies installed:

bash
Copy code
pip install tensorflow numpy matplotlib pandas scikit-learn
Usage
https://github.com/sivaprakasha-datascientist/Disease-Prediction/blob/main/README.md

cd plant-disease-prediction
Prepare your dataset in the required format. Ensure the images are labeled as bright_blind, healthy, late_blight, or early_blight.

Run the training script:

bash
Copy code
python train_model.py
To predict the disease status for a new image:
bash
Copy code
python predict_disease.py --image path_to_image
Model Performance
Accuracy: The model achieves an accuracy of approximately XX% on the test set.
Loss Function: Categorical cross-entropy loss.
Optimizer: Adam optimizer was used for training.
Contributing
Feel free to fork the repository, make improvements, or submit issues and pull requests. Contributions are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

