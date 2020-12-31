# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:52:13 2020

@author: rapha
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-darkgrid')
train_1a=pd.read_csv(r"C:\Users\rapha\Desktop\GrandeDimProjet\EEG\bsi_competition_ii_train1a.csv")

lin=np.linspace(1,train_1a.shape[1],train_1a.shape[1])
plt.plot(lin,train_1a.iloc[2,:])


channel_1=train_1a.iloc[:,1:897]
channel_1["Individuals"]=[i for i in range(0,268)]
channel_1_m=channel_1.melt(id_vars=['Individuals'], value_vars=[str(i) for i in range(1,897)])
channel_1_m["variable"]=channel_1_m["variable"].apply(int)
channel_1_m.index=channel_1_m["Individuals"].apply(str)+["_"]*len(channel_1_m)+channel_1_m["variable"].apply(str)
channel_1_m.columns=["Individuals","Time","Channel_1"]

channel_2=train_1a.iloc[:,1*896+1:2*896+1]
channel_2.columns=[str(i) for i in range(1,897)]
channel_2["Individuals"]=[i for i in range(0,268)]
channel_2_m=channel_2.melt(id_vars=['Individuals'], value_vars=[str(i) for i in range(1,897)])
channel_2_m["variable"]=channel_2_m["variable"].apply(int)
channel_2_m.index=channel_2_m["Individuals"].apply(str)+["_"]*len(channel_2_m)+channel_2_m["variable"].apply(str)
channel_2_m.columns=["Individuals","Time","Channel_2"]

channel_3=train_1a.iloc[:,2*896+1:3*896+1]
channel_3.columns=[str(i) for i in range(1,897)]
channel_3["Individuals"]=[i for i in range(0,268)]
channel_3_m=channel_3.melt(id_vars=['Individuals'], value_vars=[str(i) for i in range(1,897)])
channel_3_m["variable"]=channel_3_m["variable"].apply(int)
channel_3_m.index=channel_3_m["Individuals"].apply(str)+["_"]*len(channel_3_m)+channel_3_m["variable"].apply(str)
channel_3_m.columns=["Individuals","Time","Channel_3"]

channel_4=train_1a.iloc[:,3*896+1:4*896+1]
channel_4.columns=[str(i) for i in range(1,897)]
channel_4["Individuals"]=[i for i in range(0,268)]
channel_4_m=channel_4.melt(id_vars=['Individuals'], value_vars=[str(i) for i in range(1,897)])
channel_4_m["variable"]=channel_4_m["variable"].apply(int)
channel_4_m.index=channel_4_m["Individuals"].apply(str)+["_"]*len(channel_4_m)+channel_4_m["variable"].apply(str)
channel_4_m.columns=["Individuals","Time","Channel_4"]

channel_5=train_1a.iloc[:,4*896+1:5*896+1]
channel_5.columns=[str(i) for i in range(1,897)]
channel_5["Individuals"]=[i for i in range(0,268)]
channel_5_m=channel_5.melt(id_vars=['Individuals'], value_vars=[str(i) for i in range(1,897)])
channel_5_m["variable"]=channel_5_m["variable"].apply(int)
channel_5_m.index=channel_5_m["Individuals"].apply(str)+["_"]*len(channel_5_m)+channel_5_m["variable"].apply(str)
channel_5_m.columns=["Individuals","Time","Channel_5"]

channel_6=train_1a.iloc[:,5*896+1:6*896+1]
channel_6.columns=[str(i) for i in range(1,897)]
channel_6["Individuals"]=[i for i in range(0,268)]
channel_6_m=channel_6.melt(id_vars=['Individuals'], value_vars=[str(i) for i in range(1,897)])
channel_6_m["variable"]=channel_6_m["variable"].apply(int)
channel_6_m.index=channel_6_m["Individuals"].apply(str)+["_"]*len(channel_6_m)+channel_6_m["variable"].apply(str)
channel_6_m.columns=["Individuals","Time","Channel_6"]

channels_df=channel_1_m.join(channel_2_m["Channel_2"]).join(channel_3_m["Channel_3"]).join(channel_4_m["Channel_4"]).join(channel_5_m["Channel_5"]).join(channel_6_m["Channel_6"])

class_df=pd.DataFrame(train_1a.iloc[:,0])
class_df["Individuals"]=class_df.index.values
class_df.columns=["Class","Individuals"]

channels_df=channels_df.join(class_df["Class"],on="Individuals")

channels_df.groupby(by="Individuals").plot(x="Time",y=["Channel_1","Channel_2","Channel_3","Channel_4","Channel_5","Channel_6"],ylim=(-100,100))
sns.boxplot(x="variable",y="value",hue="Class",data=channels_df.melt(id_vars=["Individuals","Time","Class"]))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

x = channels_df.drop(["Individuals", "Time","Class"], axis=1)
y = channels_df["Class"]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state = 2)

lr=LogisticRegression()
lr_fit = lr.fit(x_train, y_train)

pred_train = lr_fit.predict(x_train)
pred_test = lr_fit.predict(x_test)
pred_tot=lr_fit.predict(x_scaled)

print("Accuracy train :", metrics.accuracy_score(y_train, pred_train))
1-sum(abs(y_train-pred_train))/len(y_train)

print("Accuracy test :", metrics.accuracy_score(y_test, pred_test))

print("Accuracy total :", metrics.accuracy_score(y, pred_tot))

lr_coef=pd.DataFrame(lr_fit.coef_,columns=x.columns).transpose()

channels_df["lr_predict"]=pred_tot
indiv_accuracy=[]
indiv_predict_max=[]
time_low=0
time_limit=128

for i in range(268):
    loc_acc=metrics.accuracy_score(channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"Class"], channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"])
    indiv_accuracy.append(loc_acc)
    indiv_predict_max_l=channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"].value_counts().index.values[channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"].value_counts().argmax()]
    indiv_predict_max.append(indiv_predict_max_l)
    # plt.figure()
    # plt.plot(channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"Time"],channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"Class"],label="Individual Class")
    # plt.plot(channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"Time"],channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"],label="Class Prediction")
    # plt.legend()
    
plt.plot(indiv_accuracy)
plt.plot(indiv_predict_max)


print("GLobal individual accuracy : ",metrics.accuracy_score(train_1a.iloc[:,0],indiv_predict_max))
#80% with all values
#69% with only the 128 first = 0.5sec

#Is there a more representative time interval ?
interval_size=256 #equivalent 1sec
global_indiv_acc=[]

for t in range(0,897-interval_size,16):
    print("Start at : ",str(t))
    time_low=t
    time_limit=t+interval_size
    indiv_accuracy=[]
    indiv_predict_max=[]
    for i in range(268):
        loc_acc=metrics.accuracy_score(channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"Class"], channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"])
        indiv_accuracy.append(loc_acc)
        indiv_predict_max_l=channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"].value_counts().index.values[channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"].value_counts().argmax()]
        indiv_predict_max.append(indiv_predict_max_l)
        
    global_indiv_acc.append(metrics.accuracy_score(train_1a.iloc[:,0],indiv_predict_max))
    
    
plt.plot(range(0,897-interval_size,16),global_indiv_acc)
print("Best interval : ",str(range(0,897-interval_size,16)[np.argmax(global_indiv_acc)])," - ",str(range(0,897-interval_size,16)[np.argmax(global_indiv_acc)]+interval_size))

#There seems to be interval with a better fit for prediction (using the max occurences rule). What if we re-fit a model on each interval and look at the overall accuracy (test & train) on global individuals class prediction ?
interval_size=128
global_indiv_acc=[]
for t in range(0,897-interval_size,16):
    print("Start at : ",str(t))
    time_low=t
    time_limit=t+interval_size
    indiv_accuracy=[]
    indiv_predict_max=[]
    
    x = channels_df.loc[(channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low)].drop(["Individuals", "Time","Class"], axis=1)
    y = channels_df.loc[(channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"Class"]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state = 2)
    
    lr=LogisticRegression()
    lr_fit = lr.fit(x_train, y_train)
    
    pred_tot=lr_fit.predict(x_scaled)
    channels_df.loc[(channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"]=pred_tot

    
    for i in range(268):
        loc_acc=metrics.accuracy_score(channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"Class"], channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"])
        indiv_accuracy.append(loc_acc)
        indiv_predict_max_l=channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"].value_counts().index.values[channels_df.loc[(channels_df["Individuals"]==i) & (channels_df["Time"]<=time_limit) & (channels_df["Time"]>time_low),"lr_predict"].value_counts().argmax()]
        indiv_predict_max.append(indiv_predict_max_l)
        
    global_indiv_acc.append(metrics.accuracy_score(train_1a.iloc[:,0],indiv_predict_max))
    
plt.plot(range(0,897-interval_size,16),global_indiv_acc)
print("Best interval : ",str(range(0,897-interval_size,16)[np.argmax(global_indiv_acc)])," - ",str(range(0,897-interval_size,16)[np.argmax(global_indiv_acc)]+interval_size))
#Approx same accuracy


#Try a prediction of the class on the whole time and for all sensors: 896*6 variables, 268 individuals : high dimension problem ?

