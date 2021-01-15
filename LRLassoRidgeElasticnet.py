# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:53:44 2021

@author: rapha
"""

import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, lasso_path, lars_path,LassoLarsCV, LassoLars, LogisticRegression,LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
cmap=plt.get_cmap('tab20c')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time
from stability_selection import StabilitySelection, plot_stability_path
#import sklearn.externals.joblib

#9 dimensions
acp_x_train1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\acp_x_train1a.csv")
acp_x_test1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\acp_x_test1a.csv")

#8 dimensions
nmf_x_train1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\nmf_x_train1a.csv")
nmf_x_test1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\nmf_x_test1a.csv")

#5376 dimensions
x_train1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\x_train1a.csv")
x_test1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\x_test1a.csv")

#y
y_train1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\y_train1a.csv")
y_test1a=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\y_test1a.csv")


#%% LASSO path

#%% Using Coordinate descent
cd_lasso_path_times=[]

######### ACP #########
start=time.time()
alphas_lasso, coefs_lasso, _ = lasso_path(acp_x_train1a, y_train1a,n_alphas=100,eps=0.001,verbose=True)
end=time.time()
cd_lasso_path_times.append(end-start)

coefs_lasso=coefs_lasso[0]

plt.figure()
neg_log_alphas_lasso = -np.log10(alphas_lasso)
for col,coef_l in enumerate(coefs_lasso):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=cmap(col),label="Coef_"+str(col))
    #name=plt.text(neg_log_alphas_lasso[-1]-0.3,coef_l[-1]+0.001,"Coef_"+str(col),color=cmap(col),fontsize=7)

plt.xlabel('-Log(lambda)')
plt.ylabel('coefficients')
plt.legend()
plt.title('Lasso Path (lambda from '+str(round(alphas_lasso.max(),2))+' to '+str(round(alphas_lasso.min(),2))+') \n PCA reduced datas')

######### NMF #########
start=time.time()
alphas_lasso, coefs_lasso, _ = lasso_path(nmf_x_train1a, y_train1a,n_alphas=100,eps=0.001,verbose=True)
end=time.time()
cd_lasso_path_times.append(end-start)

coefs_lasso=coefs_lasso[0]

plt.figure()
neg_log_alphas_lasso = -np.log10(alphas_lasso)
for col,coef_l in enumerate(coefs_lasso):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=cmap(col),label="Coef_"+str(col))
    #name=plt.text(neg_log_alphas_lasso[-1]-0.3,coef_l[-1]+0.001,"Coef_"+str(col),color=cmap(col),fontsize=7)

plt.xlabel('-Log(lambda)')
plt.ylabel('coefficients')
plt.legend()
plt.title('Lasso Path (lambda from '+str(round(alphas_lasso.max(),2))+' to '+str(round(alphas_lasso.min(),2))+') \n NMF reduced datas')


######### Full Dataset #########
start=time.time()
alphas_lasso, coefs_lasso, _ = lasso_path(x_train1a, y_train1a,n_alphas=100,eps=0.001,verbose=True)
end=time.time()
cd_lasso_path_times.append(end-start)

coefs_lasso=coefs_lasso[0]

plt.figure()
neg_log_alphas_lasso = -np.log10(alphas_lasso)
listcoef=[0,5,10,50,100,200,500,1000,2000,3000,4000,5000]
for col,coef_l in enumerate(coefs_lasso[listcoef]):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l,c=cmap(col),label="Coef_"+str(listcoef[col]))
    #name=plt.text(neg_log_alphas_lasso[-1]-0.3,coef_l[-1]+0.001,"Coef_"+str(col),color=cmap(col),fontsize=7)

plt.xlabel('-Log(lambda)')
plt.ylabel('coefficients')
plt.legend(loc="upper left")
plt.title('Lasso Path (lambda from '+str(round(alphas_lasso.max(),2))+' to '+str(round(alphas_lasso.min(),2))+') \n Full dataset')


#%% Using LARS
lars_lasso_path_times=[]

######### ACP #########
x=np.array(acp_x_train1a)
y=np.array(y_train1a)[:,0]

start=time.time()
alphas,_,coefs,n_iter = lars_path(x, y,max_iter=500,return_n_iter=True,method="lasso")
end=time.time()
lars_lasso_path_times.append(end-start)

plt.figure()
neg_log_alphas = -np.log10(alphas)
for col,coef_l in enumerate(coefs):
    l1 = plt.plot(neg_log_alphas, coef_l, c=cmap(col),label="Coef_"+str(col))
    #name=plt.text(neg_log_alphas_lasso[-1]-0.3,coef_l[-1]+0.001,"Coef_"+str(col),color=cmap(col),fontsize=7)

plt.xlabel('-Log(alphas)')
plt.ylabel('coefficients')
plt.legend()
plt.title('Lasso Path computed with '+str(n_iter)+' iterations of LARS \n PCA reduced datas')

######### NMF #########
x=np.array(nmf_x_train1a)
y=np.array(y_train1a)[:,0]

start=time.time()
alphas,_,coefs,n_iter = lars_path(x, y,max_iter=500,return_n_iter=True,method="lasso")
end=time.time()
lars_lasso_path_times.append(end-start)

plt.figure()
neg_log_alphas = -np.log10(alphas)
for col,coef_l in enumerate(coefs):
    l1 = plt.plot(neg_log_alphas, coef_l, c=cmap(col),label="Coef_"+str(col))
    #name=plt.text(neg_log_alphas_lasso[-1]-0.3,coef_l[-1]+0.001,"Coef_"+str(col),color=cmap(col),fontsize=7)

plt.xlabel('-Log(alphas)')
plt.ylabel('coefficients')
plt.legend()
plt.title('Lasso Path computed with '+str(n_iter)+' iterations of LARS \n NMF reduced datas')

######### Full Dataset #########
x=np.array(x_train1a)
y=np.array(y_train1a)[:,0]

start=time.time()
alphas,_,coefs,n_iter = lars_path(x, y,max_iter=5000,return_n_iter=True,method="lasso",verbose=True)
end=time.time()
lars_lasso_path_times.append(end-start)

plt.figure()
neg_log_alphas = -np.log10(alphas)
listcoef=[0,5,10,50,100,200,500,1000,2000,3000,4000,5000]
for col,coef_l in enumerate(coefs[listcoef]):
    #index=[10*i for i in range(int(len(neg_log_alphas)/10))]
    l1 = plt.plot(neg_log_alphas[:-1], coef_l[:-1], c=cmap(col),label="Coef_"+str(listcoef[col]))
    #name=plt.text(neg_log_alphas_lasso[-1]-0.3,coef_l[-1]+0.001,"Coef_"+str(col),color=cmap(col),fontsize=7)

plt.xlabel('-Log(lambda)')
plt.ylabel('coefficients')
plt.legend()
plt.title('Lasso Path computed with '+str(n_iter)+' iterations of LARS \n and lambda from '+str(round(alphas.max(),2))+' to '+str(round(alphas.min(),2))+' \n PCA reduced datas')


#%% Time comparison CD/LARS

labels = ['acp', 'nmf']
x = np.arange(len(labels))  
width = 0.35 

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, lars_lasso_path_times[0:1], width, label='LARS LASSO')
rects2 = ax.bar(x + width/2, cd_lasso_path_times[0:1], width, label='CD LASSO')

ax.set_ylabel('Time')
ax.set_title('Time of computation of the LASSO path for PCA and NMF reduced datas')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

width = 0.35 
labels=["full dataset"]
x = np.arange(len(labels))  
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, lars_lasso_path_times[2], width, label='LARS LASSO')
rects2 = ax.bar(x + width/2, cd_lasso_path_times[2], width, label='CD LASSO')

ax.set_ylabel('Time')
ax.set_title('Time of computation of the LASSO path for full dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#LARS is way faster especially for the full dataset

#%% General remarks on LASSO
#Although the results above are interesting to illustrate the efficiency of the two algorithm to solve LASSO problem, they are not meant to be used for classification purpose

#We will also be using LASSO to select variables through stability selection

#Then we will be using a penalized Logistic Regression:
#check the results of stability selection with this model (with LASSO penalty) and select variables
#perform a Logistic Regression without penalty on the selected variables and check the prediction results
#fit a penalized Logistic Regression model (with LASSO, Ridge, Elasticnet penalties) and check the prediction results

#%% Using LASSO to select variables through stability selection

######### Full Dataset #########
lasso_lars=LassoLars(fit_intercept=False, max_iter=5000,verbose=True,normalize=False)

x=np.array(x_train1a)
y=np.array(y_train1a)[:,0]

selector = StabilitySelection(n_jobs=-1,base_estimator=lasso_lars,lambda_name='alpha', lambda_grid=np.linspace(0.01, 0.3, 100), bootstrap_func='subsample',verbose=True,threshold=0.2,sample_fraction=0.5,n_bootstrap_iterations=20)
#threshold =0.2
selector.fit(x, y)

selected_variables = selector.get_support(indices=True)
selected_scores = selector.stability_scores_.max(axis=1)

plot_stability_path(selector)

proba = selector.stability_scores_
plt.figure(4, figsize=(10, 8))
colorStab = cycle([cmap(i) for i in range(20)])
for ind, c in zip(selected_variables, colorStab):
    ax = plt.plot(-np.log10(np.linspace(0.01, 0.3, 100)), proba[ind], c=c, label= ind)
plt.ylabel('Probas')
plt.xlabel('-log(Lambda)')
plt.title('Stability selection')
#plt.legend()

selected_variables
selected_scores
#There is no feature with a particularly great stability. There must be a very

######### NMF #########
lasso_lars=LassoLars(fit_intercept=False, max_iter=5000,verbose=True,normalize=False)

x=np.array(scaler.fit_transform(nmf_x_train1a))
y=np.array(y_train1a)[:,0]

selector = StabilitySelection(n_jobs=-1,base_estimator=lasso_lars,lambda_name='alpha', lambda_grid=np.linspace(0.01, 0.3, 100), bootstrap_func='subsample',verbose=True,threshold=0.6,sample_fraction=0.5,n_bootstrap_iterations=100)
#threshold =0.6
selector.fit(x, y)

selected_variables = selector.get_support(indices=True)
selected_variables
selected_scores = selector.stability_scores_.max(axis=1)
selected_scores

plot_stability_path(selector)

proba = selector.stability_scores_
plt.figure(4, figsize=(10, 8))
colorStab = cycle([cmap(i) for i in range(20)])
for ind, c in zip(selected_variables, colorStab):
    ax = plt.plot(-np.log10(np.linspace(0.01, 0.3, 100)), proba[ind], c=c, label= ind)
plt.ylabel('Probas')
plt.xlabel('-Log(Lambda)')
plt.title('Stability selection')
plt.legend()

#Every features are kept, they all tends to be selected with a low lambda

######### ACP #########
lasso_lars=LassoLars(fit_intercept=False, max_iter=5000,verbose=True,normalize=False)

x=np.array(scaler.fit_transform(acp_x_train1a))
y=np.array(y_train1a)[:,0]

selector = StabilitySelection(n_jobs=-1,base_estimator=lasso_lars,lambda_name='alpha', lambda_grid=np.linspace(0.01, 5, 100), bootstrap_func='subsample',verbose=True,threshold=0.6,sample_fraction=0.5,n_bootstrap_iterations=100)
#threshold =0.6
selector.fit(x, y)

selected_variables = selector.get_support(indices=True)
selected_variables
selected_scores = selector.stability_scores_.max(axis=1)
selected_scores

plot_stability_path(selector)

proba = selector.stability_scores_
plt.figure(4, figsize=(10, 8))
colorStab = cycle([cmap(i) for i in range(20)])
for ind, c in zip(selected_variables, colorStab):
    ax = plt.plot(-np.log10(np.linspace(0.01, 5, 100)), proba[ind], c=c, label= ind)
plt.ylabel('Probas')
plt.xlabel('-log(Lambda)')
plt.title('Stability selection')
plt.legend()

#all variables tends to be selected for a low lambda


#Stability selection with LASSO is not very helpful to select variables


#%% Stability selection with Logistic Regression 
#LASSO penalty only, the Ridge penalty is not meant to select variables by setting some of teir coefficients at 0, it just reduces the value of coefficients

######### Full Dataset #########
lr_lasso=LogisticRegression(penalty="l1",solver="liblinear")
#liblinear is much faster than saga solver, it seems to be appropriate for a dataset with few observations

x=np.array(x_train1a)
y=np.array(y_train1a)[:,0]

selector = StabilitySelection(n_jobs=-1,base_estimator=lr_lasso,lambda_name='C', lambda_grid=np.linspace(0.01, 1, 100), bootstrap_func='subsample',verbose=True,threshold=0.2,sample_fraction=0.9,n_bootstrap_iterations=100,random_state=1)
#threshold =0.9
selector.fit(x, y) #long to perform (100 lambdas * 100 bootstrap iterations !)


selected_variables = selector.get_support(indices=True)
selected_scores = selector.stability_scores_.max(axis=1)

plot_stability_path(selector)

proba = selector.stability_scores_
plt.figure(4, figsize=(10, 8))
colorStab = cycle([cmap(i) for i in range(20)])
len(selected_variables)
listcoef=[0,1,2,3,4,5,10,50,100,200]
for ind, c in zip(selected_variables[listcoef], colorStab):
    ax = plt.plot((np.linspace(0.01, 1, 100)), proba[ind], c=c, label= ind)
plt.ylabel('Probas')
plt.xlabel('C')
plt.title('Stability selection LR with LASSO Penalty (only some random features are shown)')
plt.legend()

highstab=(proba[:,50:]>0.8).all(axis=1) #always over 80% selection for C>0.5 (lower constraint)
#highstab=((proba>0.6).sum(axis=1)>40) #more than 60% selection for more than 40 different C
stable_indices = [i for i, x in enumerate(highstab) if x == True]
stable_indices
len(stable_indices) #only 12 stable features !

plt.figure(4, figsize=(10, 8))
colorStab = cycle([cmap(i) for i in range(20)])
#selected_variables=proba[stable_indices,:]
for ind, c in zip(stable_indices, colorStab):
    ax = plt.plot((np.linspace(0.01, 1, 100)), proba[ind], c=c, label= ind)
plt.ylabel('Probas')
plt.xlabel('C')
plt.title('Stability selection LR with LASSO Penalty (selected features with P>0.8 for C>0.5)')
plt.legend()

######### NMF #########
lr_lasso=LogisticRegression(penalty="l1",solver="liblinear",max_iter=1000)

x=np.array(scaler.fit_transform(nmf_x_train1a))
y=np.array(y_train1a)[:,0]

selector = StabilitySelection(n_jobs=-1,base_estimator=lr_lasso,lambda_name='C', lambda_grid=np.linspace(0.01, 1, 100), bootstrap_func='subsample',verbose=True,threshold=0.2,sample_fraction=0.9,n_bootstrap_iterations=20)
#threshold =0.9
selector.fit(x, y)

selected_variables = selector.get_support(indices=True)
selected_scores = selector.stability_scores_.max(axis=1)

plot_stability_path(selector)

proba = selector.stability_scores_
plt.figure(4, figsize=(10, 8))
colorStab = cycle([cmap(i) for i in range(20)])
for ind, c in zip(selected_variables, colorStab):
    ax = plt.plot((np.linspace(0.01, 1, 100)), proba[ind], c=c, label= ind)
plt.ylabel('Probas')
plt.xlabel('C')
#The higher the C the lower the penalty strength
plt.title('Stability selection : LR with LASSO Penalty, on NMF datas')
plt.legend() #every variables tends to be selected

selected_variables
selected_scores

######### ACP #########
lr_lasso=LogisticRegression(penalty="l1",solver="liblinear",max_iter=1000)

x=np.array(scaler.fit_transform(acp_x_train1a))
y=np.array(y_train1a)[:,0]

selector = StabilitySelection(n_jobs=-1,base_estimator=lr_lasso,lambda_name='C', lambda_grid=np.linspace(0.01, 1, 100), bootstrap_func='subsample',verbose=True,threshold=0.2,sample_fraction=0.9,n_bootstrap_iterations=100, random_state=1)
#threshold =0.9
selector.fit(x, y)

selected_variables = selector.get_support(indices=True)
selected_scores = selector.stability_scores_.max(axis=1)

plot_stability_path(selector)

proba = selector.stability_scores_
plt.figure(4, figsize=(10, 8))
colorStab = cycle([cmap(i) for i in range(20)])
for ind, c in zip(selected_variables, colorStab):
    ax = plt.plot((np.linspace(0.01, 1, 100)), proba[ind], c=c, label= ind)
plt.ylabel('Probas')
plt.xlabel('C')
#The higher the C the lower the penalty strength
plt.title('Stability selection')
plt.legend(loc="upper left")

(proba[:,75:]>0.99).all(axis=1)
#Features number 3 and number 8 are much less selected than others

selected_variables
selected_scores

#%% Create a dataframe with all prediction results on the 67 tests observations
pred_results=pd.DataFrame(data=y_test1a.copy())
pred_results.columns=["label"]

#%% Non penalized Logistic Regression with selected variables
lr=LogisticRegression(penalty="none",solver="lbfgs")
#liblinear can't be used for "none" penalty

######### Reduced Full Dataset #########
reduced_x_train1a=x_train1a.iloc[:,stable_indices]
reduced_x_test1a=x_test1a.iloc[:,stable_indices]

x=np.array(reduced_x_train1a)
y=np.array(y_train1a)[:,0]

lr=lr.fit(x,y)

y_true=y_test1a

start=time.time()
y_pred=lr.predict(reduced_x_test1a)
end=time.time()

end-start
accuracy_score(y_pred,y_true) 
#prediction time 0.00099s
#Accuracy 89.552%

pred_results["Reduced_Full_NoP"]=y_pred
pred_results["Acc_Reduced_Full_NoP"]=(pred_results["Reduced_Full_NoP"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_Reduced_Full_NoP"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#More accurate for the 0 class

######### NMF #########

#no further reduction to do
x=np.array(scaler.fit_transform(nmf_x_train1a))
y=np.array(y_train1a)[:,0]

lr=lr.fit(x,y)

y_true=y_test1a

start=time.time()
y_pred=lr.predict(scaler.fit_transform(nmf_x_test1a))
end=time.time()

end-start
accuracy_score(y_pred,y_true) 
#prediction time 0.004s
#Accuracy 82.09%

pred_results["NMF_NoP"]=y_pred
pred_results["Acc_NMF_NoP"]=(pred_results["NMF_NoP"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_NMF_NoP"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Equilibrate accuracy

######### Reduced ACP #########
#reduction by withdrawing features number 3 and 8
acp_stable_indices=[0,1,2,4,5,6,7]
reduced_acp_x_train1a=acp_x_train1a.iloc[:,acp_stable_indices]
reduced_acp_x_test1a=acp_x_test1a.iloc[:,acp_stable_indices]

x=np.array(scaler.fit_transform(reduced_acp_x_train1a))
y=np.array(y_train1a)[:,0]

lr=lr.fit(x,y)

y_true=y_test1a

start=time.time()
y_pred=lr.predict(scaler.fit_transform(reduced_acp_x_test1a))
end=time.time()

end-start
accuracy_score(y_pred,y_true) 
#Accuracy 85.07%
#prediction time 0.005s

pred_results["ACP_Reduced_NoP"]=y_pred
pred_results["Acc_ACP_Reduced_NoP"]=(pred_results["ACP_Reduced_NoP"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_ACP_Reduced_NoP"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Quite equilibrate accuracy

#%% Penalized Logistic Regression 

#%% LASSO penalty

######### ACP #########
lr_l1=LogisticRegressionCV(Cs=100,penalty="l1",solver="liblinear",n_jobs=-1,verbose=1,scoring='accuracy')

x=np.array(scaler.fit_transform(acp_x_train1a))
y=np.array(y_train1a)[:,0]

lr_l1.fit(x,y)

lr_l1.classes_
lr_l1.coef_ #unsurprisingly features number 3 and 8 have a 0 coefficient
lr_l1.intercept_
lr_l1.Cs_

lr_l1.scores_[1.0].max(axis=1) #best performance for each fold
np.argmax(lr_l1.scores_[1.0],axis=1) #argmax of the best performance for each fold
lr_l1.Cs_[np.argmax(lr_l1.scores_[1.0],axis=1)] #best Cs for each fold (if refit=False the best C is the average of these 5)

lr_l1.scores_[1.0].mean(axis=0) #average performance on the 5 folds for each Cs
lr_l1.Cs_[np.argmax(lr_l1.scores_[1.0].mean(axis=0))] #best C

lr_l1.C_ #best C

y_true=y_test1a

start=time.time()
y_pred=lr_l1.predict(scaler.fit_transform(acp_x_test1a))
end=time.time()

end-start
accuracy_score(y_pred,y_true) 
#Accuracy 85.07%
#prediction time 0.004s
pred_results["ACP_L1P"]=y_pred
pred_results["Acc_ACP_L1P"]=(pred_results["ACP_L1P"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_ACP_L1P"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Equilibrate accuracy

lr.coef_ #lr must be fitted with the reduced acp datas
lr_l1.coef_
#Exactly the same accuracy although coefficients are shrinked in the lr_l1


######### NMF #########
lr_l1=LogisticRegressionCV(Cs=100,penalty="l1",solver="liblinear",n_jobs=-1,verbose=1,scoring='accuracy')

x=np.array(scaler.fit_transform(nmf_x_train1a))
y=np.array(y_train1a)[:,0]

lr_l1.fit(x,y)
lr_l1.C_ #best C
lr_l1.coef_

y_true=y_test1a

start=time.time()
y_pred=lr_l1.predict(scaler.fit_transform(nmf_x_test1a))
end=time.time()

end-start
accuracy_score(y_pred,y_true) 
#Accuracy 85.07% 
#prediction time 0.004s

pred_results["NMF_L1P"]=y_pred
pred_results["Acc_NMF_L1P"]=(pred_results["NMF_L1P"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_NMF_L1P"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Equilibrate accuracy
#exact same accuracy than in the penalized logistic regression on ACP datas (corresponding to 57 correctly classified observations)
#Are they making the exact same classification ?
(pred_results["NMF_L1P"]==pred_results["ACP_L1P"]).all() 
#No

######### Full Dataset #########

lr_l1=LogisticRegressionCV(Cs=100,penalty="l1",solver="liblinear",n_jobs=-1,verbose=1,scoring='accuracy')

x=np.array(x_train1a)
y=np.array(y_train1a)[:,0]

lr_l1.fit(x,y)
lr_l1.C_ #best C
lr_l1.coef_
(lr_l1.coef_!=0).sum() #42 non 0 coefficients
(lr_l1.coef_==0).sum()/len(lr_l1.coef_[0,:]) #99.2% coefficients are 0

y_true=y_test1a

start=time.time()
y_pred=lr_l1.predict(x_test1a)
end=time.time()

end-start
accuracy_score(y_pred,y_true) 

#Accuracy 89.552% #same accuracy than the non penalized reduced Full dataset (with 12 features)
#prediction time 0.04s

pred_results["Full_L1P"]=y_pred
pred_results["Acc_Full_L1P"]=(pred_results["Full_L1P"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_Full_L1P"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Accuracy is not equilibrate, classify more in the 0 class

#%% Ridge penalty
######### ACP #########
lr_l2=LogisticRegressionCV(Cs=100,penalty="l2",solver="lbfgs",n_jobs=-1,verbose=1,scoring='accuracy')

x=np.array(scaler.fit_transform(acp_x_train1a))
y=np.array(y_train1a)[:,0]

lr_l2.fit(x,y)

lr_l2.coef_ #no coefs are set to 0

lr_l2.scores_[1.0].mean(axis=0) #average performance on the 5 folds for each Cs
lr_l2.Cs_[np.argmax(lr_l2.scores_[1.0].mean(axis=0))] #best C

lr_l2.C_ #best C, very low, high constraint

y_true=y_test1a

start=time.time()
y_pred=lr_l2.predict(scaler.fit_transform(acp_x_test1a))
end=time.time()

end-start
accuracy_score(y_pred,y_true) 
#Accuracy 85.07%
#prediction time 0.004s

pred_results["ACP_L2P"]=y_pred
pred_results["Acc_ACP_L2P"]=(pred_results["ACP_L2P"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_ACP_L2P"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Equilibrate accuracy



######### NMF #########
lr_l2=LogisticRegressionCV(Cs=100,penalty="l2",solver="lbfgs",n_jobs=-1,verbose=1,scoring='accuracy')

x=np.array(scaler.fit_transform(nmf_x_train1a))
y=np.array(y_train1a)[:,0]

lr_l2.fit(x,y)
lr_l2.C_ #best C
lr_l2.coef_

y_true=y_test1a

start=time.time()
y_pred=lr_l2.predict(scaler.fit_transform(nmf_x_test1a))
end=time.time()

end-start
accuracy_score(y_pred,y_true) 
#Accuracy 86.57% 
#prediction time 0.003s

pred_results["NMF_L2P"]=y_pred
pred_results["Acc_NMF_L2P"]=(pred_results["NMF_L2P"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_NMF_L2P"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Equilibrate accuracy


######### Full Dataset #########
lr_l2=LogisticRegressionCV(Cs=100,penalty="l2",solver="lbfgs",n_jobs=-1,verbose=1,scoring='accuracy')

x=np.array(x_train1a)
y=np.array(y_train1a)[:,0]

lr_l2.fit(x,y) #longer to fit !
lr_l2.C_ #best C
lr_l2.coef_
(lr_l2.coef_!=0).sum() #All non 0 coefficients ! Not sparse at all

y_true=y_test1a

start=time.time()
y_pred=lr_l2.predict(x_test1a)
end=time.time()

end-start
accuracy_score(y_pred,y_true) 

#Accuracy 85.07%
#prediction time 0.03s

pred_results["Full_L2P"]=y_pred
pred_results["Acc_Full_L2P"]=(pred_results["Full_L2P"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_Full_L2P"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Accuracy is not very equilibrate, classify more in the 0 class


#%% Elasticnet penalty
######### ACP #########
lr_elasticnet=LogisticRegressionCV(Cs=100,penalty="elasticnet",l1_ratios=[0.05*i for i in range(21)],solver="saga",n_jobs=-1,verbose=1,scoring='accuracy')

x=np.array(scaler.fit_transform(acp_x_train1a))
y=np.array(y_train1a)[:,0]

lr_elasticnet.fit(x,y)

lr_elasticnet.coef_ #One coefficient at 0 (feature 8, feature 3 is kept non 0)

lr_elasticnet.C_ #best C, very low, high constraint
lr_elasticnet.l1_ratio_ #low but non 0 L1 ratio, mostly a L2 penalty, but a bit of LASSO (that explains the coef at 0 for feature 8)

y_true=y_test1a

start=time.time()
y_pred=lr_elasticnet.predict(scaler.fit_transform(acp_x_test1a))
end=time.time()

end-start
accuracy_score(y_pred,y_true)
#Accuracy 83.58%
#prediction time 0.004s
#Surprising result, accuracy is lower (56 well classified) than with a Ridge or a LASSO penalty (57 well classified), thus we would expect the function to take l1_ratio=0 or 1 and achieve this better result
#NB setting l1_ratios to [0] yields to a 85.07% accuracy...


lr_manualcv=LogisticRegression(penalty="elasticnet",solver="saga",n_jobs=1,verbose=1)

grid={"C":lr_l1.Cs_.tolist(),"l1_ratio":[0.05*i for i in range(21)]}
gs=GridSearchCV(estimator=lr_manualcv,param_grid=grid,scoring="accuracy",n_jobs=-1,verbose=3)

gs.fit(x,y)
gs.best_params_
lr_best=gs.best_estimator_

y_pred=lr_best.predict(scaler.fit_transform(acp_x_test1a))
accuracy_score(y_pred,y_true)
#Same surprising result with GridSearchCV


pred_results["ACP_ElasticP"]=y_pred
pred_results["Acc_ACP_ElasticP"]=(pred_results["ACP_ElasticP"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_ACP_ElasticP"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Equilibrate accuracy

######### NMF #########
lr_elasticnet=LogisticRegressionCV(Cs=100,penalty="elasticnet",l1_ratios=[0.05*i for i in range(21)],solver="saga",n_jobs=-1,verbose=1,scoring='accuracy')

x=np.array(scaler.fit_transform(nmf_x_train1a))
y=np.array(y_train1a)[:,0]

lr_elasticnet.fit(x,y)

lr_elasticnet.coef_ #No coefficient at 0

lr_elasticnet.C_ #best C, very low, high constraint
lr_elasticnet.l1_ratio_ #0 L1 ratio, only a ridge penalty

y_true=y_test1a

start=time.time()
y_pred=lr_elasticnet.predict(scaler.fit_transform(nmf_x_test1a))
end=time.time()

end-start
accuracy_score(y_pred,y_true)
#Accuracy 86.57% #same as LR with Ridge Penalty of course
#prediction time 0.003s 

pred_results["NMF_ElasticP"]=y_pred
pred_results["Acc_NMF_ElasticP"]=(pred_results["NMF_ElasticP"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_NMF_ElasticP"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Equilibrate accuracy


######### Full Dataset #########
lr_elasticnet=LogisticRegressionCV(Cs=20,penalty="elasticnet",l1_ratios=[0.2*i for i in range(6)],solver="saga",max_iter=1000,n_jobs=-1,verbose=1,scoring='accuracy')
#reducing the number of Cs and l1_ratio because the model is too long to fit

x=np.array(x_train1a)
y=np.array(y_train1a)[:,0]

lr_elasticnet.fit(x,y) #very long to fit

lr_elasticnet.coef_ 
(lr_elasticnet.coef_!=0).sum() #all coefficients not 0

lr_elasticnet.C_ #best C, very low, high constraint
lr_elasticnet.l1_ratio_ #0 L1 ratio, only a ridge penalty

y_true=y_test1a

start=time.time()
y_pred=lr_elasticnet.predict(x_test1a)
end=time.time()

end-start
accuracy_score(y_pred,y_true)

pred_results["Full_ElasticP"]=y_pred
pred_results["Acc_Full_ElasticP"]=(pred_results["Full_ElasticP"]==pred_results["label"])
cross_accuracy_stats=pd.crosstab(pred_results["label"],pred_results["Acc_Full_ElasticP"])
cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
cross_accuracy_stats
#Quite unequilibrate accuracy classify more in the 0 class

#%%Summary of accurcies

pred_results.columns.to_list()
names=['Acc_Reduced_Full_NoP','Acc_NMF_NoP', 'Acc_ACP_Reduced_NoP','Acc_ACP_L1P', 'Acc_NMF_L1P', 'Acc_Full_L1P','Acc_ACP_L2P', 'Acc_NMF_L2P','Acc_Full_L2P', 'Acc_ACP_ElasticP','Acc_Full_ElasticP', 'Acc_NMF_ElasticP']
values=[pred_results[n].sum()/67 for n in names]

summary_acc=pd.DataFrame(data=names)
summary_acc["values"]=values
summary_acc.to_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\summary_accuracy_LR.csv",index=False)