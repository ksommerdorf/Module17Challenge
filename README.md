# Credit Risk Analysis
## Overview of Project
Fast Lending is a peer to peer leading services company who is looking to utilize machine learnig to predict credit risk. The goal is to provide a quicker and more reliable loan experience and to lead to a more accurate identification of good candidates for loans which will lead to lower default rates. Credit risk is an inherently unbalanced classification problem, as good loans outweighs risky loans. Therefore this project employed different supervised machine learnig techniques to train and evaluate models with unbalanced classes using python libraries, imablanced-learn and scikit-learn in order to determine the most accurate model.

## Results
The models are evaulated on their effectiveness based on these three scores:
* Balanced Accuracy Score: measures how accurate the model predicts credit risk.
* Precision Score: 
  * For High Risk: True Positive/(True Positive + False Positive)
  * For Low Risk: True Negative/(True Negative + False Negative)
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
### SMOTE 
Synthetic Minority Oversample Technique, like RandomOverSampler, increases the size of the minority class by synthesizes new values based on the closest existing value.

### ClusterCentroid
An algorithm that decreases the size of the majority class by generating synthetic data points, centroids, that represent clusters of the sample data.

### SMOTEEN
Synthetic Minority Oversampling Tecnique and Edited NearestNeighbors model combines aspects of both oversampling using SMOTE and undersampling by dropping out the outliers of each of the classes of data.

### BalancedRandomForestClassifier
A model that randomly undersamples each boostrap sample by creating 2 trees of the same size and equal size to the minority class to represent one for the majority class and one for the minority class. 

### EasyEnsembleClassifier
A model that builds sequences of classifiers by resampling the majority class. The classifiers are an ensembler of adaptive boosting (AdaBoost) learners trained on different balanced (through undersampling) boostrap examples.

## Summary 
