# Network Intrusion Detection System

This repository contains the implementation of various machine learning models for binary classification of network traffic into normal and abnormal (attack) categories using the KDD Cup 1999 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to build a network intrusion detection system that can distinguish between normal and abnormal network traffic. We use various machine learning algorithms to achieve this task and compare their performance.

## Dataset

We use the KDD Cup 1999 dataset, which is widely used for benchmarking network intrusion detection systems. The dataset contains a wide variety of intrusions simulated in a military network environment.

- **Training data:** `KDDTrain+.txt`
- **Test data:** `KDDTest+.txt`

## Dependencies

To run the code in this repository, you need to have the following libraries installed:

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, make_scorer, plot_roc_curve
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import scipy
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
