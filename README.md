# Predicting Job Placement Using Naïve Bayes Algorithm
## Introduction
With the increasing demand for skilled professionals, organizations continuously seek efficient recruitment strategies to identify and hire the best candidates. Traditional hiring processes often involve manual screening of applicants based on their educational qualifications, work experience, and aptitude scores. However, these methods can be time-consuming and prone to human biases.

To enhance the recruitment process, machine learning techniques can be leveraged to predict whether a candidate is likely to get placed based on their academic background, work experience, and other relevant attributes. This project aims to develop a predictive model using the Naïve Bayes algorithm, a probabilistic classifier based on Bayes’ theorem, to determine the likelihood of job placement for candidates in the given dataset.

As a beginner-friendly machine learning problem, this project provides an excellent opportunity to explore different classification techniques, evaluate their performance, and understand key challenges in predictive modeling. The approach taken in solving this problem will enhance our understanding of machine learning concepts and
can help educational institutions and recruiters gain insights into key factors influencing job placements, ultimately optimizing hiring decisions and career guidance strategies.


## Nayes Bayes Algorithm
The Naïve Bayes algorithm is a probabilistic classification technique based on Bayes' Theorem. It assumes that the features are independent of each other, making it a "naïve" assumption. Despite this simplification, Naïve Bayes performs well in many real-world applications, especially in text classification, spam detection, and medical diagnosis. The algorithm calculates the probability of each class given the input features and selects the class with the highest probability. It is computationally efficient, works well with small datasets, and is particularly useful for categorical data.

## Dataset Overview
Dataset is taken from kaggle, which is uploaded as "drug_classification.csv",link:https:https://www.kaggle.com/datasets/ahsan81/job-placement-dataset

The dataset contains various attributes related to a candidate’s academic background, work experience, and test scores, which influence their chances of getting placed.
Here’s a structured table describing the dataset and its columns:  
  

| **Column Name**         | **Description**                                      | **Data Type** |
|-------------------------|------------------------------------------------------|--------------|
| **gender**              | Gender of the candidate (Male/Female)               | Categorical  |
| **ssc_percentage**      | Senior secondary (10th Grade) exam percentage       | Numerical    |
| **ssc_board**          | Board of education for SSC exams (e.g., Central, State) | Categorical  |
| **hsc_percentage**      | Higher secondary (12th Grade) exam percentage       | Numerical    |
| **hsc_board**          | Board of education for HSC exams (e.g., Central, State) | Categorical  |
| **hsc_subject**         | Subject of study in HSC (e.g., Science, Commerce, Arts) | Categorical  |
| **degree_percentage**   | Percentage of marks in the undergraduate degree     | Numerical    |
| **undergrad_degree**    | Undergraduate degree major (e.g., Engineering, Business) | Categorical  |
| **work_experience**     | Past work experience (Yes/No)                       | Categorical  |
| **emp_test_percentage** | Employment aptitude test percentage                 | Numerical    |
| **specialization**      | Postgraduate specialization (MBA majors: HR/Marketing, Finance) | Categorical  |
| **mba_percent**         | Percentage of marks in MBA degree                   | Numerical    |
| **status (Target)**     | Placement status (Placed/Not Placed)                | Categorical  |


## **Objective of the Study**  

The objective of this project is to develop a **machine learning model using the Naïve Bayes algorithm** to predict whether a candidate will be placed or not based on their academic performance, work experience, and aptitude test scores.  

This prediction model aims to:  
- **Identify key factors influencing job placements**  
- **Provide data-driven insights for recruiters and educational institutions**  
- **Assist students in understanding how their academic and professional background affects employability**  
- **Optimize hiring decisions using machine learning techniques**  


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

### **4️⃣ Feature Scaling Using Standardization (Whether Needed)**    
- **StandardScaler** is applied to ensure all features have equal weight in distance computation.  

---
### **5️⃣ Training the Naive Bayes Model**  
- The Multinomial Naïve Bayes algorithm is applied to train the model.It works well with categorical data and is commonly used for text classification but can be adapted for discrete numerical features.
The model learns the probabilities of each feature contributing to the placement outcome.

---

### **6️⃣ Making Predictions on Test Data**  
- After training, the model predicts whether a candidate will be placed or not based on their academic and professional background.
Each test sample is classified as either "Placed" (1) or "Not Placed" (0).

---

### **7️⃣ Evaluating Model Performance**  

#### **Confusion Matrix Analysis** ,**Accuracy Score & Classification Report**  
-Assess the model’s performance using metrics such as accuracy, classification report (precision, recall, F1-score), and confusion matrix. These metrics help determine how well the model is predicting the placement status.

- The **confusion matrix** provides insights into **true positives, false positives, true negatives, and false negatives**.  
- The **accuracy score** is computed to evaluate the overall performance of the model.  
- A **classification report** is generated, providing **precision, recall, and F1-score**.  

---

## **Conclusion**  
                                    
In this project, we implemented a Multinomial Naïve Bayes classifier to predict job placement outcomes based on candidates' academic performance, work experience, and aptitude test scores. The model was trained on a dataset containing educational and professional attributes, with categorical features encoded and numerical features normalized to enhance predictive performance. After splitting the data into training and testing sets, the classifier was evaluated using accuracy, precision, recall, and F1-score.This machine learning approach provides valuable insights for recruiters and students, enabling data-driven hiring decisions and career planning.









