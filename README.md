# Sapphire-ML
An end-to-end Machine Learning pipeline that enables users to classify up to 10 class labels, using Support Vector Classifier (SVC) , Decision Tree Classifier (DTC) , Random Forest Classifier (RFC) and Extreme Gradient Boosting (XGB)

# Link
https://sapphire-ml.herokuapp.com/

## Introduction 
In order to make ML more accessible for everyone, we created _Sapphire-ML_, a machine learning pipeline. It contains the following features - 
- Data Gathering (In the form of CSV)
- Selecting the target variable
- Filling the null values
- Removing/Replacing outliers
- Shows a correlation heatmap
- Shows final Accuracy, Precision, Recall using SVC, DTC, RFC, LR, KNN & XGB 

In addition, all further reading and descriptions are added to the individual pages that will help you expand your knowledge!

## Sample Dataset 

By going to https://sapphire-ml.herokuapp.com/sample/ -- you can check out _Sapphire-ML_ without having to upload a CSV! This dataset is a pre-loaded dataset on Sapphire-ML about water potability, which can be found on Kaggle  [here](https://www.kaggle.com/adityakadiwal/water-potability). Choose `potability` as the target variable.

### Dev Notes
For developers and experts, these following parameters were used:
Backend is developed in Django, Frontend with HTML, CSS and Bootstrap 4.

**Hyper-params** : For SVC, the kernel is chosen as _rbf_, and for precision,recall - the avaerage is set as _macro_.

The reason to limit the classification labels to 10 is to restrict users from accidentally using the app for regression.

### Future Prospects
This project is still very new and hence a lot can be done further! This includes:
- Assessing Data in correct labels.
- Exploratry Data Analysis using charts and plots.
- Showing feature importance graphs to the user. 
- The ability to choose the models to use. 
- K-Cross Validation.
- Hyper-parameter Tuning. 
