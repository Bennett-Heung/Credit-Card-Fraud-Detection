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

Table 1: Descriptive statistics of each variable in the dataset

|       |     Time |               V1 |               V2 |               V3 |               V4 |               V5 |               V6 |               V7 |               V8 |               V9 |              V10 |              V11 |              V12 |              V13 |              V14 |              V15 |              V16 |              V17 |              V18 |              V19 |             V20 |              V21 |              V22 |              V23 |              V24 |            V25 |             V26 |              V27 |              V28 |      Amount |           Class |
|:------|---------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|-----------------:|-----------------:|-----------------:|-----------------:|---------------:|----------------:|-----------------:|-----------------:|------------:|----------------:|
| count | 284807   | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807           | 284807          | 284807           | 284807           | 284807           | 284807           | 284807         | 284807          | 284807           | 284807           | 284807      | 284807          |
| mean  |  94813.9 |      3.91956e-15 |      5.68817e-16 |     -8.76907e-15 |      2.78231e-15 |     -1.55256e-15 |      2.01066e-15 |     -1.69425e-15 |     -1.92703e-16 |     -3.13702e-15 |      1.76863e-15 |      9.17032e-16 |     -1.81066e-15 |      1.69344e-15 |      1.47905e-15 |      3.48234e-15 |      1.39201e-15 |     -7.52849e-16 |      4.32877e-16 |      9.04973e-16 |      5.0855e-16 |      1.53729e-16 |      7.95991e-16 |      5.36759e-16 |      4.45811e-15 |      1.453e-15 |      1.6991e-15 |     -3.66016e-16 |     -1.20605e-16 |     88.3496 |      0.00172749 |
| std   |  47488.1 |      1.9587      |      1.65131     |      1.51626     |      1.41587     |      1.38025     |      1.33227     |      1.23709     |      1.19435     |      1.09863     |      1.08885     |      1.02071     |      0.999201    |      0.995274    |      0.958596    |      0.915316    |      0.876253    |      0.849337    |      0.838176    |      0.814041    |      0.770925   |      0.734524    |      0.725702    |      0.62446     |      0.605647    |      0.521278  |      0.482227   |      0.403632    |      0.330083    |    250.12   |      0.0415272  |
| min   |      0   |    -56.4075      |    -72.7157      |    -48.3256      |     -5.68317     |   -113.743       |    -26.1605      |    -43.5572      |    -73.2167      |    -13.4341      |    -24.5883      |     -4.79747     |    -18.6837      |     -5.79188     |    -19.2143      |     -4.49894     |    -14.1299      |    -25.1628      |     -9.49875     |     -7.21353     |    -54.4977     |    -34.8304      |    -10.9331      |    -44.8077      |     -2.83663     |    -10.2954    |     -2.60455    |    -22.5657      |    -15.4301      |      0      |      0          |
| 25%   |  54201.5 |     -0.920373    |     -0.59855     |     -0.890365    |     -0.84864     |     -0.691597    |     -0.768296    |     -0.554076    |     -0.20863     |     -0.643098    |     -0.535426    |     -0.762494    |     -0.405571    |     -0.648539    |     -0.425574    |     -0.582884    |     -0.468037    |     -0.483748    |     -0.49885     |     -0.456299    |     -0.211721   |     -0.228395    |     -0.54235     |     -0.161846    |     -0.354586    |     -0.317145  |     -0.326984   |     -0.0708395   |     -0.0529598   |      5.6    |      0          |
| 50%   |  84692   |      0.0181088   |      0.0654856   |      0.179846    |     -0.0198465   |     -0.0543358   |     -0.274187    |      0.0401031   |      0.022358    |     -0.0514287   |     -0.0929174   |     -0.0327574   |      0.140033    |     -0.0135681   |      0.0506013   |      0.0480715   |      0.0664133   |     -0.0656758   |     -0.00363631  |      0.00373482  |     -0.0624811  |     -0.0294502   |      0.00678194  |     -0.0111929   |      0.0409761   |      0.0165935 |     -0.0521391  |      0.00134215  |      0.0112438   |     22      |      0          |
| 75%   | 139320   |      1.31564     |      0.803724    |      1.0272      |      0.743341    |      0.611926    |      0.398565    |      0.570436    |      0.327346    |      0.597139    |      0.453923    |      0.739593    |      0.618238    |      0.662505    |      0.49315     |      0.648821    |      0.523296    |      0.399675    |      0.500807    |      0.458949    |      0.133041   |      0.186377    |      0.528554    |      0.147642    |      0.439527    |      0.350716  |      0.240952   |      0.0910451   |      0.07828     |     77.165  |      0          |
| max   | 172792   |      2.45493     |     22.0577      |      9.38256     |     16.8753      |     34.8017      |     73.3016      |    120.589       |     20.0072      |     15.595       |     23.7451      |     12.0189      |      7.84839     |      7.12688     |     10.5268      |      8.87774     |     17.3151      |      9.25353     |      5.04107     |      5.59197     |     39.4209     |     27.2028      |     10.5031      |     22.5284      |      4.58455     |      7.51959   |      3.51735    |     31.6122      |     33.8478      |  25691.2    |      1          |


Figure 1: Class countplot 

![Class countplot](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/class.png)


Figure 2: Feature variable distributions 

![Feature variable distributions](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/distplots.png)


**Key findings**

* The number of fraudulent transactions relative to genuine transactions indicates that this dataset is severely imbalanced 
* 'Amount' distribution is skewed right 
* 'Time' has a bimodal distribution
* Most variables are distributed roughly around 0, besides fradulent classes where variable distributions are different between genuine and fraudulent classes: V3, V4, V10, V11, V12, V14, V16, V17, V18. 

Figure 3: Heatmap of correlations of each variable 

![corr_heatmap](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/corrheatmap.png)


Figure 4: Scatter plot between Time and Amount

![Time_Amount](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/Time_Amount.png)


**Key findings**
* No notable correlation between the numbered variables - from V1 to V28
* There are noticeable correlations between 'Time' and 'Amount', which are examined further in the figures below:
  * 'Time': negative correlation with V3
  * 'Amount': positive correlation with V7 and V20, and negative correlation with V2
* No correlation between time and amount for both genuine and fraudulent transactions. 

Figures 5-8: Scatter plots and line of best fits by Class

![V2_Amount](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/V2_Amount.png)


![V7_Amount](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/V7_Amount.png)


![V20_Amount](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/V20_Amount.png)


![V3_Time](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/V3_Time.png)


**Key finding**
Only observable correlations are between 'Amount' and variables V2, V7 and V20 for genuine transactions (Class=0) only. 

## Modelling
* Feature Engineering: 'Time' and 'Amount' variables were standardised (like z-scores) before modelling. 
* Linear Regression, Random Forest Classifier and XGBoost Classifier were the selected models. Note that there are other classifiers, such as Support Vector Classification and Light GBM Classifier, which were not included to restrain the time length to run the codes. 
  * Weights for each model were adjusted to accommodate for the imbalanced data. 
  * These models were baseline models; hyperparameter tuning of these models was the next step, coupled with various resampling techniques (and without resampling). 
  * Resampling techniques included Random Undersampling, Tomek Links, Random Oversampling, SMOTE and SMOTE-Tomek. 
  * Both resampling and hyperparameter tuning was involved to select the best model(s) based on the performance metrics, which are mentioned below.
* Accuracy was not an appropriate measure for imbalanced data; precision, recall, F1-score, AUROC (area under the ROC curve) and AUPRC (area under the precision-recall curve) are metrics considered to evaluate model performance. Note that optimal thresholds were not assessed during cross-validation as the area-under-curve metrics were chosen the determine the best models overall, rather at a specific threshold. 
* Stratified K-fold Cross Validation was applied, rather than nested cross-validation, to restrain the time length to run the codes below. Stratified K-folds were used rather than K-folds given that there were not many fraudulent transactions in the dataset.  

Table 2: Results of cross validating the original models (i.e. without resampling or hyperparameter tuning)

|    | model_name                    | Precision       | Recall          | F1 Score        | AUROC           | AUPRC           |
|---:|:------------------------------|:----------------|:----------------|:----------------|:----------------|:----------------|
|  0 | Logistic Regression           |0.061654         |**0.916131**     |0.115500         |0.980907         |0.753420         |
|  1 | Random Forest Classifier      |**0.951280**     |0.748588         |0.836472         |0.948944         |0.835786         |
|  2 | XGBoost Classifier            |0.913743         |0.822201         |**0.865079**     |**0.982511**     |**0.851396**     |

*Note: the scores are averages of scores on the validation sets.*

**Key Findings**
* Logisitc Regression had the highest Recall score, but returned a very low Precision score, F1 score and AUPRC.
* Random Forest Classifier had the highest Precision score.
* XGBoost Classifier has the highest Recall score, F1 score, AUROC and AUPRC.
* XGBoost Classifier is the preferred (baseline) model - consistent with top scores; a steady recall without the cost of significantly less precision and vice versa.    

Table 3: Results of the top models after resampling and hyperparameter tuning

|    | model_name                                      | Precision       | Recall          | F1 Score        | AUROC           | AUPRC           |
|---:|:------------------------------------------------|:----------------|:----------------|:----------------|:----------------|:----------------|
|  0 | Tuned XGBoost Classifier (without resampling)   |**0.958042**     |0.789192         |0.864752         |0.982509         |**0.852028**     |
|  1 | XGBoost Classifier with Random Oversampling     |0.923269         |0.819669         |**0.868017**     |**0.982739**     |0.849754         |
|  2 | XGBoost Classifier with SMOTE-Tomek             |0.788984         |**0.827264**     |0.807288         |0.977423         |0.844292         |

*Note: the scores are averages of scores on the validation sets and the combination of hyperparameters for each XGBoost Classifier are different.*

**Key Findings**
* Tuned XGBoost Classifier (without resampling) - higher precision and AUPRC but at the cost of a signifcantly lower recall score.
*  XGBoost Classifier with Random Oversampling - slightly higher F1 score and AUROC, but slightly lower recall score than the original XGB Classifier.
*  XGBoost Classifier with SMOTE-Tomek - a slightly better recall score at a great cost of lower precision. 

The top two models based on the results above are in the table below. 

Table 4: Best models 
|    | model_name                                    | Precision       | Recall          | F1 Score        | AUROC           | AUPRC           |
|---:|:----------------------------------------------|:----------------|:----------------|:----------------|:----------------|:----------------|
|  0 | Original XGBoost Classifier                   |0.913743         |**0.822201**     |0.865079         |0.982511         |**0.851396**     |
|  1 | XGBoost Classifier with Random Oversampling   |**0.923269**     |0.819669         |**0.868017**     |**0.982739**     |0.849754         |

*Note: the scores are averages of scores on the validation sets and the combination of hyperparameters for each XGBoost Classifier are different.*

The results above show both models have similar scores. However, the original XGBoost Classifier (without resampling or tuning) is the best model between the two. 
* Recall has a greater focus than precision, as the cost of incorrectly not detecting fraud transactions (predicted Class=0, true Class=1) is greater than incorrectly detecting genuine transactions (predicted Class=1, true Class=0). 
* AUROC is typically high for imbalanced data and this happens to be the case throughout the modelling process. Thus, it does not hold much weight as an AUC (area-under-curve) measure in comparison to AUPRC. 
* The F1 score, as a harmonic mean between precision and recall. The original XGBoost had a slightly lower F1 score than the random oversampling XGBoost Classifier. This is due to the result of the greater difference between precision scores relative to the difference between recall scores. 
* From these findings, the original XGBoost Classifier is the best model, as well the more attractive one in terms of the less time it takes for it to run. 


## Model evaluation 
Table 5: Classification report of best model on test data

|              | Precision    | Recall    | F1 Score    |Support       |
|-------------:|:-------------|:----------|:------------|:-------------|
|Genuine       | 1.00         |1.00       |1.00         |56864         |
|Fraudulent    | 0.87         |0.85       |0.86         |98            |
|    |          |       |         |            |
|Accuracy    |          |       |1.00         |56962            |
|Macro Average    | 0.94         |0.92       |0.93         |56962            |
|Weighted Average    | 1.00         |1.00       |1.00         |56962            |


Figures 9: Confusion matrix of best model on test data

![creditcard_conf_matrix](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/creditcard_conf_matrix.png)


Figures 10 and 11: ROC Curve and Precision-Recall Curve for the best model (on test data)

![creditcard_curves](https://github.com/Bennett-Heung/Credit-Card-Fraud-Detection/blob/main/images/creditcard_curves.png)


The evaluation of the model on the test data above showed similar results to the k-fold cross validation. This gives confidence in deploying our selected model. 

* The code for the best model is: XGBClassifier(random_state=random_seed, scale_pos_weight = weights, use_label_encoder=False), where: 
  * random_seed = 42
  * weights = the total number of Class=0 / the total number of Class=1
  * Rest of the hyperparameters are default. 
