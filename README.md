# Predicting Job Placement Using Naïve Bayes Algorithm
## Introduction
With the increasing demand for skilled professionals, organizations continuously seek efficient recruitment strategies to identify and hire the best candidates. Traditional hiring processes often involve manual screening of applicants based on their educational qualifications, work experience, and aptitude scores. However, these methods can be time-consuming and prone to human biases.

To enhance the recruitment process, machine learning techniques can be leveraged to predict whether a candidate is likely to get placed based on their academic background, work experience, and other relevant attributes. This project aims to develop a predictive model using the Naïve Bayes algorithm, a probabilistic classifier based on Bayes’ theorem, to determine the likelihood of job placement for candidates in the given dataset.

## Objective

The goal of this project is to develop a Naïve Bayes classification model that can predict whether a candidate will be placed or not based on their academic background, work experience, and test performance. The model will be evaluated using standard performance metrics such as accuracy, precision, recall, and F1-score to assess its effectiveness.

As a beginner-friendly machine learning problem, this project provides an excellent opportunity to explore different classification techniques, evaluate their performance, and understand key challenges in predictive modeling for healthcare applications. The approach taken in solving this problem will enhance our understanding of machine learning concepts and
can help educational institutions and recruiters gain insights into key factors influencing job placements, ultimately optimizing hiring decisions and career guidance strategies.


## Nayes Bayes Algorithm
The Naïve Bayes algorithm is a probabilistic classification technique based on Bayes' Theorem. It assumes that the features are independent of each other, making it a "naïve" assumption. Despite this simplification, Naïve Bayes performs well in many real-world applications, especially in text classification, spam detection, and medical diagnosis. The algorithm calculates the probability of each class given the input features and selects the class with the highest probability. It is computationally efficient, works well with small datasets, and is particularly useful for categorical data.
## Dataset Overview
Dataset is taken from kaggle, which is uploaded as "drug_classification.csv",link:https:https://www.kaggle.com/datasets/ahsan81/job-placement-dataset

##  **Step-by-Step Implementation of Naive Bayes Algorithm**

### **1️⃣ Data Loading & Preprocessing**  
- **Reading the dataset** and inspecting its structure.  
- **Checking for missing values**   
- **Encoding categorical variables**
- **A pie chart or barchart is drawn using matplotlib to visualize the distribution of categorical variables in the dataset.**

---

### **2️⃣ Splitting Data into Features & Target Variable**  
- **Defining the feature matrix (x)**, which includes all predictor variables.  
- **Defining the target variable (y)**, which contains the type of drug required.

---

### **3️⃣ Splitting Data into Training & Testing Sets**  
- **Dividing the dataset** into **70% training data and 30% testing data** using stratified sampling to ensure an even distribution of classes.  

---

### **4️⃣ Feature Scaling Using Standardization**    
- **StandardScaler** is applied to ensure all features have equal weight in distance computation.  

---

### **5️⃣ Training the Naive Bayes Model**  
- Use the Bernoulli Naïve Bayes classifier, which is suitable for binary feature datasets. Train the model using the training data, where it learns the probability distribution of each binary feature in relation to different drug classifications. The model then applies Bayes' Theorem to predict the most probable drug type for new patient data.
---

### **6️⃣ Making Predictions on Test Data**  
- Once the model is trained, use it to predict drug types for the test data. The model assigns probabilities to each drug type and selects the one with the highest probability.  

---

### **7️⃣ Evaluating Model Performance**  

#### **Confusion Matrix Analysis** ,**Accuracy Score & Classification Report**  
-Assess the model’s performance using metrics such as accuracy, classification report (precision, recall, F1-score), and confusion matrix. These metrics help determine how well the model is classifying different drug types.
- The **confusion matrix** provides insights into **true positives, false positives, true negatives, and false negatives**.  
- The **accuracy score** is computed to evaluate the overall performance of the model.  
- A **classification report** is generated, providing **precision, recall, and F1-score** for each milk quality.  

---

## **Conclusion**  
                                    
In this project, we successfully implemented a **Naïve Bayes classifier** to predict drug types based on patient attributes such as **age, sex, blood pressure, cholesterol levels, and sodium-to-potassium ratio**.By using Bernoulli Naïve Bayes we effectively handled both numerical and categorical data, achieving a reliable classification model. The model’s performance was evaluated using accuracy, precision, recall, and a confusion matrix, demonstrating its effectiveness in drug classification. This project highlights the power of **machine learning in healthcare**, offering a data-driven approach to assist in medication recommendations. 
