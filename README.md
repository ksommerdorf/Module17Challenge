# Credit Risk Analysis
## Overview of Project
Fast Lending is a peer to peer leading services company who is looking to utilize machine learnig to predict credit risk. The goal is to provide a quicker and more reliable loan experience and to lead to a more accurate identification of good candidates for loans which will lead to lower default rates. Credit risk is an inherently unbalanced classification problem, as good loans outweighs risky loans. Therefore this project employed different supervised machine learnig techniques to train and evaluate models with unbalanced classes using python libraries, imablanced-learn and scikit-learn in order to determine the most accurate model.

## Results
### RandomOverSampler
Minority classes are randomly selected and added to the training set until majority and minority outcomes are equal.

### SMOTE 
Synthetic Minority Oversample Technique, like RandomOverSampler, increases the size of the minority class by synthesizes new values based on the closest existing value.

### ClusterCentroi 
An algorithm that decreases the size of the majority class by generating synthetic data points, centroids, that represent clusters of the sample data.

### SMOTEEN
Synthetic Minority Oversampling Tecnique and Edited NearestNeighbors model combines aspects of both oversampling using SMOTE and undersampling by dropping out the outliers of each of the classes of data.

### BalancedRandomForestClassifier
A model that randomly undersamples each boostrap sample by creating 2 trees of the same size and equal size to the minority class to represent one for the majority class and one for the minority class. 

### EasyEnsembleClassifier
A model that builds sequences of classifiers by resampling the majority class. The classifiers are an ensembler of adaptive boosting (AdaBoost) learners trained on different balanced (through undersampling) boostrap examples.

## Summary 
