# Project 1: [Project Title]
**Team Members**:  
Amélie Menoud (330315)
Lisa van de Panne  (395822)
Nathan Kabas Kuroiwa (341429)  

**Aicrowd Team Name**: Team_LAN

## Overview
This project focuses on developing and evaluating multiple machine learning models to estimate the likelihood of developing coronary heart disease, given a dataset from the BRFSS surveys. We explore several approaches including logistic regression, support vector machines, and other methods, with detailed hyperparameter tuning and model evaluation techniques.

## Repository Contents
In this repository, you will find the following files (📌 indicates mandatory deliverables):

- 📌 **`README.md`**: This readme file
- 📌 **`implementations.py`**: Contains the required functions for Step 2 of the assignment, as well as additional machine learning methods used in other models.
- **`implementations_notebook.ipynb`**: A notebook that demonstrates the 6 methods of Step 2 applied to the dataset.
- **`log_reg.ipynb`**: A notebook with a detailed implementation of regularized logistic regression, including hyperparameter tuning using K-fold cross-validation and grid search.
- **`SVM.ipynb`**: A notebook detailing our Support Vector Machine model, with hyperparameter tuning using K-fold cross-validation and grid search.
- 📌 **`run.py`**: Contains the code to reproduce our best submission on AICrowd.
- **`utilities`**: A folder with additional code files used to train our models:
  - **`Data_preprocessing_global.py`**: Includes tools for data preprocessing, such as data transformation for specific features, standardization, and imputation.
  - **`helpers.py`**: Tools to load data, create AICrowd submissions, and evaluate model performance.
  - **`Hyperparameters_Logreg.py`** and **`Hyperparameters_SVM.py`**: Functions for running cross-validation and selecting optimal hyperparameters.

## Remarks
In the notebooks, when loading the data, the path to the folder containing such data must be re-specified by the user, according to your where it is stored locally.
