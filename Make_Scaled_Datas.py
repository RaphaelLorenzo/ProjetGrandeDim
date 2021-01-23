# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:24:04 2020

@author: rapha
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-darkgrid')
train_1a=pd.read_csv(r"C:\Users\rapha\Desktop\GrandeDimProjet\EEG\bsi_competition_ii_train1a.csv")
test_1a=pd.read_csv(r"C:\Users\rapha\Desktop\GrandeDimProjet\EEG\bsi_competition_ii_test1a.csv")

scaler = StandardScaler()

scaled_test_1a=pd.DataFrame(scaler.fit_transform(test_1a))
t=pd.DataFrame(scaler.fit_transform(train_1a.iloc[:,1:]))
scaled_train_1a=pd.DataFrame(train_1a["0"]).join(t)

scaled_test_1a.to_csv(r"C:\Users\rapha\Desktop\GrandeDimProjet\scaled_test_1a.csv",index=False)
scaled_train_1a.to_csv(r"C:\Users\rapha\Desktop\GrandeDimProjet\scaled_train_1a.csv",index=False)

