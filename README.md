# Credit-Card-Fraud-Detection
Recognising fraudulent credit card transactions.

This repository contains: 
* 'Credit_Card_Fraud_Detection.ipynb' - notebook file
* 'creditcard.zip' - zip folder contains the raw data
* 'Images' folder - contains the image outputs of the notebook file
* 'README.md' - summary report for the repository.

## Objective
Not all credit card transactions are genuine, as some are actually fraudulent. Fraudulent transactions are not only illicit, but costly to credit card lenders, such as banks, and customers. The objective is to detect fraudulent credit card transactions so customers are not charged for items that they actually did not purchase. Relative to the total number of credit card transactions made, fraudulent transactions are significantly less frequent. This is reflected in the imbalanced dataset, sourced from [here](https://www.kaggle.com/mlg-ulb/creditcardfraud), containing credit card transactions by european cardholders in September 2013. 

## Data
The dataset is imbalanced, with the positive class (frauds) account comprising only 0.172% of all transactions (492 frauds within 284,807 transactions). The variables in the dataset are: 
* Feature 'Class' - the target variable and it takes value 1 in case of fraud and 0 otherwise
* The feature variables, 'V1, V2, … V28', are the principal components obtained with PCA - transformed due to confidentiality issues 
* 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset
* 'Amount' - the transaction amount. 

## Data Preparation
* The features variables were floats and the target variable 'Class' were integers 
* No null observations were detected 
* Only 'Time' and 'Amount' variables were standardised before modelling. 

## Exploratory Data Analysis (EDA)
|       |     Time |               V1 |               V2 |               V3 |               V4 |               V5 |
