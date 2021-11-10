# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:47:31 2021

@author: sophi
"""

# data origin : https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

import pandas as pd
df = pd.read_csv('C:/Users/sophi/OneDrive/Desktop/Applied Health Informatics/AHI FALL 21/healthcare-dataset-stroke-data.csv')
df

"""
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient 
"""

descriptive= df.describe()


## Variables of interest for 1- way Anovas: 
## dependent variables of interest(continuous) :heart_disease , avg_glucose_level, hypertension,  bmi,  stroke
## Independent variables: work_type ( 4 Levels) ,  smoking_status (4 levels) 

import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, bartlett
from statsmodels.formula.api import ols
import statsmodels.api as sm

## Creating box plots 
import seaborn as sns
glucose_work_boxplot = sns.boxplot(x='work_type', y= 'avg_glucose_level', data=df, palette="Set3")
bmi_work_boxplot = sns.boxplot(x='work_type', y= 'bmi', data=df, palette="Set3")
hypertension_work_boxplot = sns.boxplot(x='work_type', y= 'hypertension', data=df, palette="Set3")
hypertension_smoking_status_boxplot = sns.boxplot(x='smoking_status', y= 'hypertension', data=df, palette="Set3")
glucose_smoking_status_boxplot = sns.boxplot(x='smoking_status', y= 'avg_glucose_level', data=df, palette="Set3")
bmi_smoking_status_boxplot = sns.boxplot(x='smoking_status', y= 'bmi', data=df, palette="Set3")

## ## TEST 1:is a difference between the average glucose levels and work type groups? 
model = ols('avg_glucose_level ~ C(work_type)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=1)
anova_table
"""
 df        sum_sq       mean_sq          F        PR(>F)
C(work_type)     4.0  1.346151e+05  33653.778023  16.612273  1.558660e-13
Residual      5105.0  1.034190e+07   2025.838291        NaN           NaN
""" 
## There is no significant difference between the average glucose levels and work type groups 


## ## TEST 2:is a difference between the average glucose levels and smoking_status type groups?
model = ols('avg_glucose_level ~ C(smoking_status)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=1)
anova_table
"""
   df        sum_sq       mean_sq          F        PR(>F)
C(smoking_status)     3.0  1.128697e+05  37623.246154  18.536355  5.885157e-12
Residual           5106.0  1.036365e+07   2029.700323        NaN           NaN
"""
## There is no significant difference between the average glucose levels and smoking_status type groups 

## ## TEST3:is a difference between the bmi and smoking_status type groups?
model = ols('bmi ~ C(smoking_status)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=1)
anova_table
"""
                       df         sum_sq      mean_sq          F        PR(>F)
C(smoking_status)     3.0   22523.940045  7507.980015  131.41449  6.947847e-82
Residual           4905.0  280232.735421    57.132056        NaN           NaN
"""
## ## There is no significant difference between the bmi and smoking_status type groups 

