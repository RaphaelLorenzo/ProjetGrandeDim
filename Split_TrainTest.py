# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:06:55 2021

@author: rapha
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
 
train_1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\scaled_train_1a.csv")
y=train_1a.iloc[:,0]
X=train_1a.iloc[:,1:]

x_train1a, x_test1a, y_train1a, y_test1a= train_test_split(X, y, test_size=0.25, random_state=1)

x_train1a.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\x_train1a.csv",index=False)
x_test1a.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\x_test1a.csv",index=False)
y_train1a.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\y_train1a.csv",index=False)
y_test1a.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\y_test1a.csv",index=False)

nmf_train_1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\nmf_scaled_train_1a.csv")
X_nmf=nmf_train_1a.iloc[:,1:]

nmf_x_train1a, nmf_x_test1a, nmf_y_train1a, nmf_y_test1a= train_test_split(X_nmf, y, test_size=0.25, random_state=1)

nmf_x_train1a.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\nmf_x_train1a.csv",index=False)
nmf_x_test1a.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\nmf_x_test1a.csv",index=False)
nmf_y_train1a.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\nmf_y_train1a.csv",index=False)
nmf_y_test1a.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\nmf_y_test1a.csv",index=False)
