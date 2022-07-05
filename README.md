# Credit Risk Analysis
## Overview of Project
Fast Lending is a peer-to-peer leading services company who is looking to utilize machine learning to predict credit risk. The goal is to provide a quicker and more reliable loan experience and to lead to a more accurate identification of good candidates for loans leading to lower default rates. Credit risk is an inherently unbalanced classification problem, as good loans outweigh risky loans. Therefore, this project employed different supervised machine learning techniques to train and evaluate models with unbalanced classes using python libraries, imbalanced-learn and scikit-learn in order to determine the most accurate model.

## Results
The models are evaluated on their effectiveness based on these four scores:
* Balanced Accuracy Score: measures how accurate the model predicts credit risk
* Precision Score: 
  * For High Risk: True Positive/(True Positive + False Positive)
  * For Low Risk: True Negative/(True Negative + False Negative
  
![example1](https://user-images.githubusercontent.com/57520471/177226605-c34a4d10-5241-49e5-a646-aaa8339f2816.png)
  
* Recall (or Sensitivity) Score: 
  * For High Risk: True Positive/(True Positive + False Negative)
  * For Low Risk: True Negative/(True Negative + False Positive)
* F1 Score (harmonic mean): 2(Precision * Recall)/(Precision + Recall)
  * Best score is 1.0 and the worst score is 0.0. 

### Naive Random Oversampling 
RandomOverSampler randomly selects minority classes and adds them to the training set until majority and minority outcomes are equal.
* Balanced Accuracy Score: 64.4%
* Precision Score:  
  * High Risk: 0.01 (1% of the predicted high risk applicants are actually high risk)
  * Low Risk: 1.00 (100% of the predicted low risk applicants are actually low risk)
* Recall Score: 
  * High Risk: 0.69 (69% of high risk applicants are classified as high risk)
  * Low Risk: 0.59 (59% of low risk applicants are classified as low risk)
* F1 Score: 
  * High Risk: 0.02
  * Low Risk: 0.74
  
![1 1](https://user-images.githubusercontent.com/57520471/177226637-851ec4ea-21ae-4f0d-84cb-0148021379bb.png)
![1 2](https://user-images.githubusercontent.com/57520471/177226638-f5a840a5-e761-4bf0-b965-7bf093ea9733.png)
![1 3](https://user-images.githubusercontent.com/57520471/177226639-5d94690d-18ba-4531-88fa-e116059e23c9.png)

### SMOTE 
Synthetic Minority Oversample Technique, like RandomOverSampler, increases the size of the minority class by synthesizes new values based on the closest existing value.
* Balanced Accuracy Score: 66.3%
* Precision Score:  
  * High Risk: 0.01 (1% of the predicted high risk applicants are actually high risk)
  * Low Risk: 1.00 (100% of the predicted low risk applicants are actually low risk)
* Recall Score: 
  * High Risk: 0.63 (63% of high risk applicants are classified as high risk)
  * Low Risk: 0.69 (69% of low risk applicants are classified as low risk)
* F1 Score: 
  * High Risk: 0.02
  * Low Risk: 0.82
 
![2 1](https://user-images.githubusercontent.com/57520471/177226678-5156eecb-9929-430d-b6d8-ce5e1b2b112b.png)
![2 2](https://user-images.githubusercontent.com/57520471/177226686-fbebe737-5abe-41d5-919f-e8834505c0ad.png)
![2 3](https://user-images.githubusercontent.com/57520471/177226690-1a7e4f2a-e422-4d11-a58d-4baa7139182d.png)

### ClusterCentroid
An algorithm that decreases the size of the majority class by generating synthetic data points, centroids, that represent clusters of the sample data.
* Balanced Accuracy Score: 54.5%
* Precision Score:  
  * High Risk: 0.01 (1% of the predicted high risk applicants are actually high risk)
  * Low Risk: 1.00 (100% of the predicted low risk applicants are actually low risk)
* Recall Score: 
  * High Risk: 0.69 (69% of high risk applicants are classified as high risk)
  * Low Risk: 0.40 (40% of low risk applicants are classified as low risk)
* F1 Score: 
  * High Risk: 0.01
  * Low Risk: 0.57
  
![3 1](https://user-images.githubusercontent.com/57520471/177226720-e35cece0-4b25-4059-9d47-8ffe5a6905d8.png)
![3 2](https://user-images.githubusercontent.com/57520471/177226722-89fbe3a8-9d3e-47f3-88e5-203ed844ee03.png)
![3 3](https://user-images.githubusercontent.com/57520471/177226723-acd60e73-5cf0-4cb0-8bd0-e3f773327283.png)

### SMOTEEN
Synthetic Minority Oversampling Technique and Edited Nearest Neighbors model combines aspects of both oversampling using SMOTE and undersampling by dropping out the outliers of each of the classes of data.
* Balanced Accuracy Score: 64.5%
* Precision Score:  
  * High Risk: 0.01 (1% of the predicted high risk applicants are actually high risk)
  * Low Risk: 1.00 (100% of the predicted low risk applicants are actually low risk)
* Recall Score: 
  * High Risk: 0.72 (72% of high risk applicants are classified as high risk)
  * Low Risk: 0.57 (57% of low risk applicants are classified as low risk)
* F1 Score: 
  * High Risk: 0.02
  * Low Risk: 0.72
  
![4 1](https://user-images.githubusercontent.com/57520471/177226747-a890a12c-6522-4c9a-8da8-cf424ac50bde.png)
![4 2](https://user-images.githubusercontent.com/57520471/177226749-bcf9f829-d2a5-4d00-8ae8-b0892aefabd2.png)
![4 3](https://user-images.githubusercontent.com/57520471/177226750-13ce2d83-bebd-4c69-8c90-b7712903f1b1.png)

### BalancedRandomForestClassifier
A model that randomly undersamples each bootstrap sample by creating 2 trees of the same size and equal size to the minority class to represent one for the majority class and one for the minority class. 
* Balanced Accuracy Score: 78.8%
* Precision Score:  
  * High Risk: 0.04 (4% of the predicted high risk applicants are actually high risk)
  * Low Risk: 1.00 (100% of the predicted low risk applicants are actually low risk)
* Recall Score: 
  * High Risk: 0.67 (67% of high risk applicants are classified as high risk)
  * Low Risk: 0.91 (91% of low risk applicants are classified as low risk)
* F1 Score: 
  * High Risk: 0.07
  * Low Risk: 0.95
  
![5 1](https://user-images.githubusercontent.com/57520471/177226767-97c6dcce-5fac-4d20-a322-5ca7c2154ec8.png)
![5 2](https://user-images.githubusercontent.com/57520471/177226769-f4b6fd53-1406-4a60-87ff-4c96b90694a7.png)
![5 3](https://user-images.githubusercontent.com/57520471/177226771-08dd8259-fc4d-4162-b27b-1b1172cbe6fd.png)

### EasyEnsembleClassifier
A model that builds sequences of classifiers by resampling the majority class. The classifiers are an ensembler of adaptive boosting (AdaBoost) learners trained on different balanced (through undersampling) bootstrap examples.
* Balanced Accuracy Score: 92.5%
* Precision Score:  
  * High Risk: 0.07 (7% of the predicted high risk applicants are actually high risk)
  * Low Risk: 1.00 (100% of the predicted low risk applicants are actually low risk)
* Recall Score: 
  * High Risk: 0.91 (91% of high risk applicants are classified as high risk)
  * Low Risk: 0.94 (94% of low risk applicants are classified as low risk)
* F1 Score: 
  * High Risk: 0.14
  * Low Risk: 0.97
  
![6 1](https://user-images.githubusercontent.com/57520471/177226784-c5cf6eb8-f3cc-42fd-9974-eaccd90ea7b6.png)
![6 2](https://user-images.githubusercontent.com/57520471/177226785-04ccb739-66f5-4a63-b45b-41ce833eb0f3.png)
![6 3](https://user-images.githubusercontent.com/57520471/177226787-c9a2467a-83e1-426f-acde-1cf5d2589e84.png)

## Summary 
Ranking of models from most accurate to least accurate for identifying high risk candidates:
* EasyEnsembleClassifer: 92.5% accuracy, 7% precision, 91% recall, and 14% F1 Score
* BalancedRandomForestClassifer: 78.8% accuracy, 4% precision, 67% recall and 7% F1 Score
* SMOTE: 66.3% accuracy, 1% precision, 63% recall and 2% F1 Score
* SMOTEENN: 64.5% accuracy, 1% precision, 72% recall and 2% F1 Score
* RandomOverSampler: 64.4% accuracy, 1% precision, 69% recall and 2% F1 Score
* ClusterCentroids: 54.5% accuracy, 1% precision, 69% recall and 1% F1 Score

After evaluating all six techniques, the EasyEnsembleClassifier has the highest accuracy score (92.5%) and the highest precision score (7%) and recall score (91%) for identifying high risk candidates. Therefore, using the EasyEnsembleClassifier is the recommended algorithm for the credit card data set. However, the downside to the model is that disportionately more candidates are classified as high risk (91%) versus being an actual high risk (7%). Since credit card companies would rather classify low risk candidates as high risk versus high risk candidates as low risk the downside is not significant enough to rule out the algorithm. The downside can also be mediated by creating a separate algorithm that will further sort through the candidates that are identified as high risk in order to further rule out low risk candidates. 
