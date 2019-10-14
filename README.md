# jp_morgan_wdbc
assignment on WDBC open source dataset

## Project Overview
Predicting whether a diagnosed breast cancer tumor cell is malignant or benign based on Wisconsin dataset from UCI repository.

## To achieve this goal, the following steps are identified: 
• Familiarize with the data by looking at its shape, the relations between variables and their possible correlations. 
• Preprocess data 
• Split the data into testing and training samples 
• Implement various classifiers (K-Nearest Neighbors, Losgistic Regression, Lasso Logistic Regression and Random Forest) to predict the data  
  via 10-fold Cross-Validation 
• Compare the best identified classifier with evaluation metric: Accuracy, as this is the one initially reported in the data description 
  file
• Write conclusions


## Libraries used
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import time
