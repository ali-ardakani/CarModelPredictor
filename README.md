# Car Model Prediction with Machine Learning

## Overview
This repository contains a machine learning project designed to assign titles of car advertisements to their respective car models. The project employs various data preprocessing techniques, data balancing, and machine learning models to achieve an accuracy of approximately 80% in predicting 1000 car models.

## Project Motivation
### Data Collection
The backbone of this project is the dataset, meticulously curated through web scraping from Divar, a prominent Iranian online marketplace. Our dataset's origin from real-world car advertisements adds a layer of authenticity and complexity that simulates the challenges of a practical use case.

### Data Preprocessing
The initial phase of our project was dedicated to meticulous data preprocessing. To ensure high-quality data, we employed the Hazm library for comprehensive text normalization and rigorous punctuation removal. However, the road ahead was not without its hurdles.

### Data Balancing
One of the most formidable challenges we faced was the substantial data imbalance inherent in our dataset. A creative solution was devised to address this issue. Car model names with fewer than a thousand samples were identified as underrepresented classes. From each of these models, some random subsample was drawn. To augment the dataset's complexity and test the model's robustness, we introduced between 1 and 5 noise words into the titles. This introduced noise simulated real-world variations and added an extra layer of challenge for the machine learning model.

### Text Vectorization
Transforming the textual data into a numerical format was paramount. We initially experimented with Bag of Words (BoW) with binary encoding due to the removal of additional words. After giving noise to the titles , we soon realized that the diversity in the wording of car titles required a more sophisticated approach. The Term Frequency-Inverse Document Frequency (TF-IDF) technique was subsequently employed. This technique not only captures word presence but also assigns importance to words based on their significance in the text.

### Model Selection
The heart of the project lay in the selection of an appropriate machine learning algorithm. We conducted an exhaustive evaluation of various algorithms, including Random Forest, Decision Trees, Support Vector Machines (SVM), and Stochastic Gradient Descent (SGD) classification. Ultimately, our choice to employ SGD classification was guided by several crucial factors:

* Large Class Count: With a staggering 1000 car models to predict, SGD classification demonstrated superior scalability in handling a vast number of classes when compared to other algorithms.

* Computational Efficiency: Given the considerable dataset size, computational efficiency was a paramount consideration. SGD classification emerged as the optimal choice, offering an impressive balance between model performance and computational speed.

* Robustness: Introducing noise into the dataset by adding random words to the titles necessitated a model that could navigate and interpret this added complexity. SGD classification demonstrated a remarkable ability to handle this noise while making informed predictions.

### Model Performance
Our chosen SGD classification model achieved an accuracy of about 80% and an F1 score of approximately 81% in predicting car models based on advertisement titles. It's important to note that the quality of data and preprocessing methods can affect the model's performance.

## Accessing the Dataset
The primary dataset for this project is available on Kaggle. It originates from car advertisements on Divar. You can download it from the following Kaggle link: [Kaggle Dataset Link](https://www.kaggle.com/datasets/aliardakani78/caradtitles-modelpredict)

## Acknowledgments
If you find this project useful or have suggestions, please feel free to contribute. We appreciate your support and welcome any feedback.

## License
This project is licensed under the MIT License. Please refer to the LICENSE file for usage and distribution details.

