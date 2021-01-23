# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:34:49 2021

@author: rapha
"""

project_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\"
data_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

y_test1a=pd.read_csv(data_path+r"y_test1a.csv")
acp_x_test1a=pd.read_csv(data_path+r"acp_x_test1a.csv")

# #Creating the general DataFame of results
# model_names=["label",
#               "Logistic_ReducedFull_NoP",
#               "Logistic_NMF_NoP",
#               "Logistic_ReducedACP_NoP",
#               "Logistic_ACP_L1P",
#               "Logistic_NMF_L1P",
#               "Logistic_Full_L1P",
#               "Logistic_ACP_L2P",
#               "Logistic_NMF_L2P",
#               "Logistic_Full_L2P",
#               "Logistic_ACP_ElasticP",
#               "Logistic_NMF_ElasticP",
#               "Logistic_Full_ElasticP",
#               "Logistic_Full_SCGroupP",
#               "Logistic_Full_TimeGroupP",
#               "Logistic_Full_ChannelGroupP",
#               "SVC_ACP_rbfKernel",
#               "SVC_ACP_linearKernel",
#               "SVC_NMF_rbfKernel",
#               "SVC_NMF_linearKernel",
#               "SVC_Full_rbfKernel",
#               "SVC_Full_linearKernel",
#               "SVC_ACP_Best",
#               "SVC_NMF_Best",
#               "SVC_Full_Best",
#               "RF_ACP_Best",
#               "RF_NMF_Best",
#               "RF_Full_Best",
#               "BaggingKN_ACP_Best",
#               "BaggingKN_NMF_Best",
#               "BaggingKN_Full_Best",
#               "AdaBoostDT_ACP_Best",
#               "AdaBoostDT_NMF_Best",
#               "AdaBoostDT_Full_Best"]

# results_pred=pd.DataFrame(np.ones((67,len(model_names))))
# results_pred.columns=model_names
# results_pred["label"]=y_test1a

# results_pred.to_csv(project_path+r"Models_Test_Results.csv",index=False)

results_pred=pd.read_csv(project_path+r"Models_Test_Results.csv")

name=[]
accuracy=[]
accuracy_0=[]
accuracy_1=[]
accuracy_gap_0_1=[]

for col in results_pred.columns[1:]:
    name.append(col)
    print(col)
    accurate=(results_pred["label"]==results_pred[col])
    acc_loc=accurate.sum()/67
    print(acc_loc)
    accuracy.append(acc_loc)
    cross_accuracy_stats=pd.crosstab(results_pred["label"],accurate)
    cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
    print(cross_accuracy_stats)
    accuracy_0.append(cross_accuracy_stats["accuracy"][0])
    accuracy_1.append(cross_accuracy_stats["accuracy"][1])
    accuracy_gap_0_1.append(abs(cross_accuracy_stats["accuracy"][0]-cross_accuracy_stats["accuracy"][1]))

dic={"Name":name,"Accuracy":accuracy,"Accuracy Class 0":accuracy_0,"Accuracy Class 1":accuracy_1,"Accuracy Gap 0-1":accuracy_gap_0_1}

stat_results=pd.DataFrame(dic)
stat_results.to_csv(project_path+"Models_Test_Results_Stats.csv")

accurate_df=results_pred.copy()

for col in results_pred.columns[1:]:
    accurate=(results_pred["label"]==results_pred[col])
    accurate_df[col]=accurate

accurate_df=accurate_df.iloc[:,1:]
sum_accurate=accurate_df.sum(axis=1)/33 #percentage of classifiers that give a good prediction


cmap = plt.cm.viridis
legend_elements = [Line2D([0], [0], marker="X",color='w', markerfacecolor="k",  markersize=10,label="True label 0"), 
                   Line2D([0], [0], marker="o",color='w', markerfacecolor="k",  markersize=10,label="True label 1")]

plt.figure(figsize=(10,6))
plt.scatter(acp_x_test1a[results_pred["label"]==0].iloc[:,0],acp_x_test1a[results_pred["label"]==0].iloc[:,1],c=sum_accurate[results_pred["label"]==0],cmap="viridis",marker="x")
plt.scatter(acp_x_test1a[results_pred["label"]==1].iloc[:,0],acp_x_test1a[results_pred["label"]==1].iloc[:,1],c=sum_accurate[results_pred["label"]==1],cmap="viridis",marker="o")
plt.xlabel("ACP Dim 1")
plt.ylabel("ACP Dim 2")
plt.title("Percentage of classifiers that give a good prediction among 33 classifiers")
plt.legend(handles=legend_elements)
cbar = plt.colorbar()

cmap = plt.cm.coolwarm
legend_elements = [Line2D([0], [0], marker="o",color='w', markerfacecolor=cmap(0.),  markersize=10,label="0"), 
                   Line2D([0], [0], marker="o",color='w', markerfacecolor=cmap(1.),  markersize=10,label="1")]

plt.figure(figsize=(10,6))
plt.scatter(acp_x_test1a.iloc[:,0],acp_x_test1a.iloc[:,1],c=results_pred["label"],cmap="coolwarm")
plt.xlabel("ACP Dim 1")
plt.ylabel("ACP Dim 2")
plt.title("Labels of test datas")
plt.legend(handles=legend_elements)

plt.figure(figsize=(10,6))
plt.scatter(acp_x_test1a.iloc[:,0],acp_x_test1a.iloc[:,1],c=results_pred["Logistic_Full_L1P"],cmap="coolwarm")
plt.xlabel("ACP Dim 1")
plt.ylabel("ACP Dim 2")
plt.title("Predicted Labels by the best classifier \n (Logistic Regression on the Full Dataset with LASSO Penalty)")
plt.legend(handles=legend_elements)


