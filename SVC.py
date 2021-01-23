# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:48:28 2021

@author: User
"""
#%% IMPORT
project_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\"
results_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\results_SVC\\"
data_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib.lines import Line2D
import time
from mpl_toolkits.mplot3d import Axes3D


#Importation données ACP
acp_x_test1a = pd.read_csv(data_path+r'acp_x_test1a.csv')
acp_x_train1a = pd.read_csv(data_path+r'acp_x_train1a.csv')
acp_y_test1a = pd.read_csv(data_path+r'acp_y_test1a.csv')
acp_y_train1a = pd.read_csv(data_path+r'acp_y_train1a.csv')
x_acp = pd.concat([acp_x_train1a, acp_x_test1a])
y_acp = pd.concat([acp_y_train1a, acp_y_test1a])


#Importation données NMF
nmf_x_test1a = pd.read_csv(data_path+r'nmf_x_test1a.csv')
nmf_x_train1a = pd.read_csv(data_path+r'nmf_x_train1a.csv')
nmf_y_test1a = pd.read_csv(data_path+r'nmf_y_test1a.csv')
nmf_y_train1a = pd.read_csv(data_path+r'nmf_y_train1a.csv')
x_nmf = pd.concat([nmf_x_train1a, nmf_x_test1a])
y_nmf = pd.concat([nmf_y_train1a, nmf_y_test1a])


#Importation données totales
x_test1a = pd.read_csv(data_path+r'x_test1a.csv')
x_train1a = pd.read_csv(data_path+r'x_train1a.csv')
y_test1a = pd.read_csv(data_path+r'y_test1a.csv')
y_train1a = pd.read_csv(data_path+r'y_train1a.csv')
x_no_red = pd.concat([x_train1a, x_test1a])
y_no_red = pd.concat([y_train1a, y_test1a])


#Results DataFrame
results_pred=pd.read_csv(project_path+r"Models_Test_Results.csv")
results_pred.columns

#%% DRAW
#Nuage de points des x train

plt.scatter(acp_x_train1a.iloc[:, 0], acp_x_train1a.iloc[:, 1], c=acp_y_train1a.iloc[:, 0], s=10, cmap='coolwarm')
plt.title('Données réduites par ACP');


plt.scatter(nmf_x_train1a.iloc[:, 0], nmf_x_train1a.iloc[:, 1], c=nmf_y_train1a.iloc[:, 0], s=10, cmap='coolwarm')
plt.title('Données réduites par NMF');


plt.scatter(x_train1a.iloc[:, 1], x_train1a.iloc[:, 0], c=y_train1a.iloc[:, 0], s=10, cmap='coolwarm')
plt.title('Données non réduites');

cmap = plt.cm.viridis
legend_elements = [Line2D([0], [0], marker="o",color='w', markerfacecolor=cmap(0.),  markersize=5,label="0"), 
                   Line2D([0], [0], marker="o",color='w', markerfacecolor=cmap(1.),  markersize=5,label="1")]


#%% SVC AVEC PARAMETRES PAR DEFAUT
#Linear kernel is mostly used when there are a Large number of Features
#rbf is used when there is no prior knowledge about the data
kernel = ["linear", "rbf"]
name_dt = ["ACP", "NMF", "No Reduction"]
x_train = [acp_x_train1a, nmf_x_train1a, x_train1a]
x_test = [acp_x_test1a, nmf_x_test1a, x_test1a]
y_train = [acp_y_train1a.iloc[:, 0], nmf_y_train1a.iloc[:, 0], y_train1a.iloc[:, 0]]
y_test = [acp_y_test1a.iloc[:, 0], nmf_y_test1a.iloc[:, 0], y_test1a.iloc[:, 0]]
accuracy_train = []
accuracy_test =[]
kern=[]
type_reduce=[]
duree=[]
hyperparam=[]

predictions=[]

hyperparam_choice = "Par défaut"
for i in range(3):
    for ker in kernel:
        print(i)
        svc=SVC(kernel=ker, probability=True) 
        debut = time.time()
        svc_fit=svc.fit(x_train[i], y_train[i])
        fin = time.time()
        temps = fin - debut
        pred_train=svc_fit.predict(x_train[i])
        pred_test=svc_fit.predict(x_test[i])
        predictions.append(pred_test)
        score_train = metrics.accuracy_score(y_train[i], pred_train)
        score_test = metrics.accuracy_score(y_test[i], pred_test)
        accuracy_train.append(score_train)
        accuracy_test.append(score_test)
        kern.append(ker)
        type_reduce.append(name_dt[i])
        hyperparam.append(hyperparam_choice)
        duree.append(temps)
        print("number of support vectors for class 0 : "+str(svc_fit.n_support_[0]))
        print("number of support vectors for class 0 : "+str(svc_fit.n_support_[1]))

results_pred["SVC_ACP_linearKernel"]=predictions[0]
results_pred["SVC_ACP_rbfKernel"]=predictions[1]
results_pred["SVC_NMF_linearKernel"]=predictions[2]
results_pred["SVC_NMF_rbfKernel"]=predictions[3]
results_pred["SVC_Full_linearKernel"]=predictions[4]
results_pred["SVC_Full_rbfKernel"]=predictions[5]
        
#%%DRAW

#2D graph
dict_graph={}

for i in range(3):
    for ker in kernel:
        svc=SVC(kernel=ker, probability=True)
        svc_fit=svc.fit(x_train[i].iloc[:,0:2], y_train[i])
        pred_test=svc_fit.predict(x_test[i].iloc[:,0:2])
        probas=svc_fit.predict_proba(x_test[i].iloc[:,0:2])
        ecart=abs(probas[:,0]-probas[:,1])
        missclass = abs(pred_test-y_test[i])
        dict_graph["pred_"+name_dt[i]+"_"+ker]=pred_test
        dict_graph["missclass_"+name_dt[i]+"_"+ker]=missclass
        dict_graph["ecart_"+name_dt[i]+"_"+ker]=ecart

#Class prediction
plt.figure(figsize=(22, 12))        
plt.subplot(321)
plt.title("Class prediction : ACP linear")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c=dict_graph["pred_ACP_linear"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(323)
plt.title("Class prediction : NMF linear")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c=dict_graph["pred_NMF_linear"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(325)
plt.title("Class prediction : No Reduction linear")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c=dict_graph["pred_No Reduction_linear"], s=10, cmap='viridis');
plt.legend(handles=legend_elements) 

plt.subplot(322)
plt.title("Class prediction : ACP rbf")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c=dict_graph["pred_ACP_rbf"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(324)
plt.title("Class prediction : NMF rbf")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c=dict_graph["pred_NMF_rbf"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(326)
plt.title("Class prediction : No Reduction rbf")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c=dict_graph["pred_No Reduction_rbf"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)    
    
    
    
#Missclassified   
plt.figure(figsize=(22, 12))        
plt.subplot(321)
plt.title("Missclassified : ACP linear")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c=dict_graph["missclass_ACP_linear"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(323)
plt.title("Missclassified : NMF linear")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c=dict_graph["missclass_NMF_linear"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(325)
plt.title("Missclassified : No Reduction linear")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c=dict_graph["missclass_No Reduction_linear"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(322)
plt.title("Missclassified : ACP rbf")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c=dict_graph["missclass_ACP_rbf"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(324)
plt.title("Missclassified : NMF rbf")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c=dict_graph["missclass_NMF_rbf"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)

plt.subplot(326)
plt.title("Missclassified : No Reduction rbf")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c=dict_graph["missclass_No Reduction_rbf"], s=10, cmap='viridis');
plt.legend(handles=legend_elements)  
    
    
#Ecart de probabilité

plt.figure(figsize=(22, 12))        
plt.subplot(321)
plt.title("Ecart de probabilité : ACP linear")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c=dict_graph["ecart_ACP_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(323)
plt.title("Ecart de probabilité : NMF linear")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c=dict_graph["ecart_NMF_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(325)
plt.title("Ecart de probabilité : No Reduction linear")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c=dict_graph["ecart_No Reduction_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(322)
plt.title("Ecart de probabilité : ACP rbf")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c=dict_graph["ecart_ACP_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(324)
plt.title("Ecart de probabilité : NMF rbf")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c=dict_graph["ecart_NMF_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(326)
plt.title("Ecart de probabilité : No Reduction rbf")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c=dict_graph["ecart_No Reduction_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()       
    

#%% GridSearch for SVC
########## GRIDSEARCH ########### 
#%%  GridSearch for SVC with PCA Datas
parameters = {'kernel':['linear', 'rbf'], 'C':np.arange(0.01,2,0.05), 
              "gamma":["auto","scale"]}

#ACP
svc = SVC(probability=True)
clf = GridSearchCV(svc, parameters,verbose=1,n_jobs=-1,scoring="accuracy")
debut = time.time()
clf_fit=clf.fit(x_train[0], y_train[0])
fin = time.time()
temps = fin - debut
acp_estimator_grid=clf_fit.best_estimator_  
print(clf_fit.best_params_)
print(clf_fit.best_score_)

#with gamma and C from 0.1 to 10 we get
#{'C': 0.1, 'gamma': 0.1, 'kernel': 'linear'}
#lowest gamma and lowest C selected, lower the choice of parameters

hyperparam_choice = "Best estimator ACP"
kernel = 'linear'

debut = time.time()
svc_fit=acp_estimator_grid.fit(x_train[0], y_train[0])
fin = time.time()
temps = fin - debut
pred_train=svc_fit.predict(x_train[0])
pred_test=svc_fit.predict(x_test[0])

results_pred["SVC_ACP_Best"]=pred_test

score_train = metrics.accuracy_score(y_train[0], pred_train)
score_test = metrics.accuracy_score(y_test[0], pred_test)
print(score_test)
accuracy_train.append(score_train)
accuracy_test.append(score_test)
kern.append(kernel)
type_reduce.append(name_dt[0])
hyperparam.append(hyperparam_choice)
duree.append(temps)

#%% DRAW
#2D Graph
svc_fit=acp_estimator_grid.fit(x_train[0].iloc[:,0:2], y_train[0])
pred_train=svc_fit.predict(x_train[0].iloc[:,0:2])
pred_test=svc_fit.predict(x_test[0].iloc[:,0:2])
probas=svc_fit.predict_proba(x_test[0].iloc[:,0:2])
ecart=abs(probas[:,0]-probas[:,1])
missclass = abs(pred_test-y_test[0])
  
support = [0 for i in range(x_train[0].shape[0])]

for i in list(svc_fit.support_):
    support[i] = 1  



plt.figure(figsize=(25, 12))        
plt.subplot(321)
plt.title("Class prediction on train sample : ACP linear")
plt.scatter(x_train[0].iloc[:, 0], x_train[0].iloc[:, 1], c=pred_train, s=10, cmap='viridis')
plt.legend(handles=legend_elements)


plt.subplot(322)
plt.title("Support vectors")
plt.scatter(x_train[0].iloc[:, 0], x_train[0].iloc[:, 1], c=support, s=10, cmap='viridis')
plt.legend(handles=legend_elements)


plt.subplot(323)
plt.title("Class prediction on test sample : ACP linear")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c = pred_test, s=10, cmap='viridis')
plt.legend(handles=legend_elements)

plt.subplot(324)
plt.title("Missclassified : ACP linear")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c=missclass, s=10, cmap='viridis')
plt.legend(handles=legend_elements)

plt.subplot(325)
plt.title("Ecart de probabilité : ACP linear")
plt.scatter(x_test[0].iloc[:, 0], x_test[0].iloc[:, 1], c=ecart, s=10, cmap='viridis_r')
cbar = plt.colorbar()


#%%  GridSearch for SVC with NMF Datas
#NMF
svc = SVC(probability=True)
clf = GridSearchCV(svc, parameters,verbose=1,n_jobs=-1,scoring="accuracy")
debut = time.time()
clf_fit=clf.fit(x_train[1], y_train[1])
fin = time.time()
temps = fin - debut
nmf_estimator_grid=clf_fit.best_estimator_
print(clf_fit.best_params_)
print(clf_fit.best_score_)


hyperparam_choice = "Best estimator NMF"
kernel = 'rbf'
debut = time.time()
svc_fit=nmf_estimator_grid.fit(x_train[1], y_train[1])
fin = time.time()
temps = fin - debut
pred_train=svc_fit.predict(x_train[1])
pred_test=svc_fit.predict(x_test[1])

results_pred["SVC_NMF_Best"]=pred_test

score_train = metrics.accuracy_score(y_train[1], pred_train)
score_test = metrics.accuracy_score(y_test[1], pred_test)
print(score_test)
accuracy_train.append(score_train)
accuracy_test.append(score_test)
kern.append(kernel)
type_reduce.append(name_dt[1])
hyperparam.append(hyperparam_choice)
duree.append(temps)

#%% DRAW
#2D graph
nmf_estimator_grid=clf_fit.best_estimator_
svc_fit=nmf_estimator_grid.fit(x_train[1].iloc[:,0:2], y_train[1])
pred_train=svc_fit.predict(x_train[1].iloc[:,0:2])
pred_test=svc_fit.predict(x_test[1].iloc[:,0:2])
probas=svc_fit.predict_proba(x_test[1].iloc[:,0:2])
ecart=abs(probas[:,0]-probas[:,1])
missclass = abs(pred_test-y_test[1])


support = [0 for i in range(x_train[1].shape[0])]
for i in list(svc_fit.support_):
    support[i] = 1  



plt.figure(figsize=(25, 12))        
plt.subplot(321)
plt.title("Class prediction on train sample : NMF linear")
plt.scatter(x_train[1].iloc[:, 0], x_train[1].iloc[:, 1], c=pred_train, s=10, cmap='viridis')
plt.legend(handles=legend_elements)


plt.subplot(322)
plt.title("Support vectors")
plt.scatter(x_train[1].iloc[:, 0], x_train[1].iloc[:, 1], c=support, s=10, cmap='viridis')
plt.legend(handles=legend_elements)


plt.subplot(323)
plt.title("Class prediction on test sample : NMF linear")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c = pred_test, s=10, cmap='viridis')
plt.legend(handles=legend_elements)

plt.subplot(324)
plt.title("Missclassified : NMF linear")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c=missclass, s=10, cmap='viridis')
plt.legend(handles=legend_elements)

plt.subplot(325)
plt.title("Ecart de probabilité : NMF linear")
plt.scatter(x_test[1].iloc[:, 0], x_test[1].iloc[:, 1], c=ecart, s=10, cmap='viridis_r')
cbar = plt.colorbar()


#%%  GridSearch for SVC with Full Datas
#No Reduction
svc = SVC(probability=True)
clf = GridSearchCV(svc, parameters,verbose=1,n_jobs=-1)
debut = time.time()
clf_fit=clf.fit(x_train[2], y_train[2])
fin = time.time()
temps = fin - debut
no_red_estimator_grid=clf_fit.best_estimator_  
print(clf_fit.best_params_)
print(clf_fit.best_score_)


hyperparam_choice = "Best estimator No Reduction"
kernel = 'linear'
debut = time.time()
svc_fit=no_red_estimator_grid.fit(x_train[2], y_train[2])
fin = time.time()
temps = fin - debut
pred_train=svc_fit.predict(x_train[2])
pred_test=svc_fit.predict(x_test[2])

results_pred["SVC_Full_Best"]=pred_test

score_train = metrics.accuracy_score(y_train[2], pred_train)
score_test = metrics.accuracy_score(y_test[2], pred_test)
print(score_test)
accuracy_train.append(score_train)
accuracy_test.append(score_test)
kern.append(kernel)
type_reduce.append(name_dt[2])
hyperparam.append(hyperparam_choice)
duree.append(temps)

#%%DRAW
#2D graph
no_red_estimator_grid=clf_fit.best_estimator_ 
svc_fit=no_red_estimator_grid.fit(x_train[2].iloc[:,0:2], y_train[2])
pred_train=svc_fit.predict(x_train[2].iloc[:,0:2])
pred_test=svc_fit.predict(x_test[2].iloc[:,0:2])
probas=svc_fit.predict_proba(x_test[2].iloc[:,0:2])
ecart=abs(probas[:,0]-probas[:,1])
missclass = abs(pred_test-y_test[2])

support = [0 for i in range(x_train[2].shape[0])]
for i in list(svc_fit.support_):
    support[i] = 1  



plt.figure(figsize=(25, 12))        
plt.subplot(321)
plt.title("Class prediction on train sample : No Reduction linear")
plt.scatter(x_train[2].iloc[:, 0], x_train[2].iloc[:, 1], c=pred_train, s=10, cmap='viridis')
plt.legend(handles=legend_elements)

plt.subplot(322)
plt.title("Support vectors")
plt.scatter(x_train[2].iloc[:, 0], x_train[2].iloc[:, 1], c=support, s=10, cmap='viridis')
plt.legend(handles=legend_elements)

plt.subplot(323)
plt.title("Class prediction on test sample : No Reduction linear")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c = pred_test, s=10, cmap='viridis')
plt.legend(handles=legend_elements)

plt.subplot(324)
plt.title("Missclassified : No Reduction linear")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c=missclass, s=10, cmap='viridis')
plt.legend(handles=legend_elements)

plt.subplot(325)
plt.title("Ecart de probabilité : No Reduction linear")
plt.scatter(x_test[2].iloc[:, 0], x_test[2].iloc[:, 1], c=ecart, s=10, cmap='viridis_r')
cbar = plt.colorbar()

#%%
results_pred.to_csv(project_path+r"Models_Test_Results.csv",index=False)


#%% Export results
dictionnaire = {"parametres" : hyperparam, "methode_reduction" : type_reduce, "kernel" : kern, 
                "accuracy_train" : accuracy_train, "accuracy_test" : accuracy_test,
                "duree" : duree}
svc_score = pd.DataFrame(dictionnaire)

svc_score.to_csv(results_path+r'summary_accuracy_SVC_1.csv', index = False) 

