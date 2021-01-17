# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 15:41:24 2021

@author: -
"""
project_path=r"C:\\Users\\rapha\\Desktop\\ProjetGrandeDIm_Local\\"
results_path=r"C:\\Users\\rapha\\Desktop\\ProjetGrandeDIm_Local\\results_SVC\\"


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib.lines import Line2D
import time

pd.set_option ('display.max_row', 12)
pd.set_option ('display.max_column', 12)
plt.style.use('seaborn-darkgrid')

#Importation données ACP
acp_x_test1a = pd.read_csv(project_path+r'acp_x_test1a.csv')
acp_x_train1a = pd.read_csv(project_path+r'acp_x_train1a.csv')
acp_y_test1a = pd.read_csv(project_path+r'acp_y_test1a.csv')
acp_y_train1a = pd.read_csv(project_path+r'acp_y_train1a.csv')
x_acp = pd.concat([acp_x_train1a, acp_x_test1a])
y_acp = pd.concat([acp_y_train1a, acp_y_test1a])


#Importation données NMF
nmf_x_test1a = pd.read_csv(project_path+r'nmf_x_test1a.csv')
nmf_x_train1a = pd.read_csv(project_path+r'nmf_x_train1a.csv')
nmf_y_test1a = pd.read_csv(project_path+r'nmf_y_test1a.csv')
nmf_y_train1a = pd.read_csv(project_path+r'nmf_y_train1a.csv')
x_nmf = pd.concat([nmf_x_train1a, nmf_x_test1a])
y_nmf = pd.concat([nmf_y_train1a, nmf_y_test1a])


#Importation données totales
x_test1a = pd.read_csv(project_path+r'x_test1a.csv')
x_train1a = pd.read_csv(project_path+r'x_train1a.csv')
y_test1a = pd.read_csv(project_path+r'y_test1a.csv')
y_train1a = pd.read_csv(project_path+r'y_train1a.csv')
x_no_red = pd.concat([x_train1a, x_test1a])
y_no_red = pd.concat([y_train1a, y_test1a])



#Nuage de points des x train

plt.scatter(acp_x_train1a.iloc[:, 0], acp_x_train1a.iloc[:, 1], c=acp_y_train1a.iloc[:, 0], s=10, cmap='coolwarm')
plt.title('Données réduites par ACP');


plt.scatter(nmf_x_train1a.iloc[:, 0], nmf_x_train1a.iloc[:, 1], c=nmf_y_train1a.iloc[:, 0], s=10, cmap='coolwarm')
plt.title('Données réduites par NMF');


plt.scatter(x_train1a.iloc[:, 1], x_train1a.iloc[:, 0], c=y_train1a.iloc[:, 0], s=10, cmap='coolwarm')
plt.title('Données non réduites');

cmap = plt.cm.coolwarm
legend_elements = [Line2D([0], [0], marker="o",color='w', markerfacecolor=cmap(0.),  markersize=5,label="0 (B)"), 
                   Line2D([0], [0], marker="o",color='w', markerfacecolor=cmap(1.),  markersize=5,label="1 (M)")]
######### PARAMETRES PAR DEFAUT
#Linear kernel is mostly used when there are a Large number of Features
#rbf is used when there is no prior knowledge about the data
kernel = ["linear", "rbf"]
name_dt = ["ACP", "NMF", "No Reduction"]
x_train = [acp_x_train1a, nmf_x_train1a, x_train1a]
x_test = [acp_x_test1a, nmf_x_test1a, x_test1a]
y_train = [acp_y_train1a.iloc[:, 0], nmf_y_train1a.iloc[:, 0], y_train1a.iloc[:, 0]]
y_test = [acp_y_test1a.iloc[:, 0], nmf_y_test1a.iloc[:, 0], y_test1a.iloc[:, 0]]
x=[x_acp, x_nmf, x_no_red]
y=[y_acp.iloc[:, 0], y_nmf.iloc[:, 0], y_no_red.iloc[:, 0]]
accuracy_train = []
accuracy_test =[]
kern=[]
type_reduce=[]
duree=[]
hyperparam=[]
dict_graph={}

hyperparam_choice = "Par défaut"
for i in range(3):
    for ker in kernel:
        svc=SVC(kernel=ker, probability=True) 
        debut = time.time()
        svc_fit=svc.fit(x_train[i], y_train[i])
        fin = time.time()
        temps = fin - debut
        pred_train=svc_fit.predict(x_train[i])
        pred_test=svc_fit.predict(x_test[i])
        pred=svc_fit.predict(x[i])
        score_train = metrics.accuracy_score(y_train[i], pred_train)
        score_test = metrics.accuracy_score(y_test[i], pred_test)
        accuracy_train.append(score_train)
        accuracy_test.append(score_test)
        kern.append(ker)
        type_reduce.append(name_dt[i])
        hyperparam.append(hyperparam_choice)
        duree.append(temps)
        probas=svc_fit.predict_proba(x[i])
        ecart=abs(probas[:,0]-probas[:,1])
        missclass = abs(pred-y[i])
        dict_graph["pred_"+name_dt[i]+"_"+ker]=pred
        dict_graph["missclass_"+name_dt[i]+"_"+ker]=missclass
        dict_graph["ecart_"+name_dt[i]+"_"+ker]=ecart
        
        
        
dictionnaire = {"parametres" : hyperparam_choice, "methode_reduction" : type_reduce, "kernel" : kern, 
                "accuracy_train" : accuracy_train, "accuracy_test" : accuracy_test,
                "duree" : duree}
svc_score = pd.DataFrame(dictionnaire)
#svc_graph = pd.DataFrame(dict_graph)


#Class prediction
plt.figure(figsize=(22, 12))        
plt.subplot(321)
plt.title("Class prediction : ACP linear")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph["pred_ACP_linear"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)

plt.subplot(323)
plt.title("Class prediction : NMF linear")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph["pred_NMF_linear"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)

plt.subplot(325)
plt.title("Class prediction : No Reduction linear")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph["pred_No Reduction_linear"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements) 

plt.subplot(322)
plt.title("Class prediction : ACP rbf")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph["pred_ACP_rbf"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)

plt.subplot(324)
plt.title("Class prediction : NMF rbf")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph["pred_NMF_rbf"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)

plt.subplot(326)
plt.title("Class prediction : No Reduction rbf")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph["pred_No Reduction_rbf"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)    
    
    
    
#Missclassified   
plt.figure(figsize=(22, 12))        
plt.subplot(321)
plt.title("Missclassified : ACP linear")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph["missclass_ACP_linear"], s=10, cmap='viridis_r');


plt.subplot(323)
plt.title("Missclassified : NMF linear")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph["missclass_NMF_linear"], s=10, cmap='viridis_r');


plt.subplot(325)
plt.title("Missclassified : No Reduction linear")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph["missclass_No Reduction_linear"], s=10, cmap='viridis_r');


plt.subplot(322)
plt.title("Missclassified : ACP rbf")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph["missclass_ACP_rbf"], s=10, cmap='viridis_r');


plt.subplot(324)
plt.title("Missclassified : NMF rbf")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph["missclass_NMF_rbf"], s=10, cmap='viridis_r');


plt.subplot(326)
plt.title("Missclassified : No Reduction rbf")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph["missclass_No Reduction_rbf"], s=10, cmap='viridis_r');
   
    
    
#Ecart de probabilité

plt.figure(figsize=(22, 12))        
plt.subplot(321)
plt.title("Ecart de probabilité : ACP linear")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph["ecart_ACP_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(323)
plt.title("Ecart de probabilité : NMF linear")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph["ecart_NMF_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(325)
plt.title("Ecart de probabilité : No Reduction linear")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph["ecart_No Reduction_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(322)
plt.title("Ecart de probabilité : ACP rbf")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph["ecart_ACP_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(324)
plt.title("Ecart de probabilité : NMF rbf")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph["ecart_NMF_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(326)
plt.title("Ecart de probabilité : No Reduction rbf")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph["ecart_No Reduction_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()       
    
    
    
########## GRIDSEARCH ########### 
parameters = {'kernel':['linear', 'rbf'], 'C':np.arange(0.1,10,0.9), 
              "gamma":np.arange(0.1,10,0.9)}

#ACP
svc = SVC(probability=True)
clf = GridSearchCV(svc, parameters,verbose=1,n_jobs=-1)
debut = time.time()
clf_fit=clf.fit(x_train[0], y_train[0])
fin = time.time()
temps = fin - debut
acp_estimator_grid=clf_fit.best_estimator_  

#NMF
svc = SVC(probability=True)
clf = GridSearchCV(svc, parameters,verbose=1,n_jobs=-1)
debut = time.time()
clf_fit=clf.fit(x_train[1], y_train[1])
fin = time.time()
temps = fin - debut
nmf_estimator_grid=clf_fit.best_estimator_

#No Reduction
svc = SVC(probability=True)
clf = GridSearchCV(svc, parameters,verbose=1,n_jobs=-1)
debut = time.time()
clf_fit=clf.fit(x_train[2], y_train[2])
fin = time.time()
temps = fin - debut
no_red_estimator_grid=clf_fit.best_estimator_     


#Best estimator ACP et No reduction   
accuracy_train = []
accuracy_test =[]
kern=[]
type_reduce=[]
duree=[]
hyperparam=[]
dict_graph_grid_acp={}

hyperparam_choice = "Best estimator ACP and No Reduction"
kernel = 'linear'
for i in range(3): 
    print(name_dt[i])
    debut = time.time()
    svc_fit=acp_estimator_grid.fit(x_train[i], y_train[i])
    fin = time.time()
    temps = fin - debut
    pred_train=svc_fit.predict(x_train[i])
    pred_test=svc_fit.predict(x_test[i])
    pred=svc_fit.predict(x[i])
    score_train = metrics.accuracy_score(y_train[i], pred_train)
    score_test = metrics.accuracy_score(y_test[i], pred_test)
    accuracy_train.append(score_train)
    accuracy_test.append(score_test)
    kern.append(kernel)
    type_reduce.append(name_dt[i])
    hyperparam.append(hyperparam_choice)
    duree.append(temps)
    probas=svc_fit.predict_proba(x[i])
    ecart=abs(probas[:,0]-probas[:,1])
    missclass = abs(pred-y[i])
    dict_graph_grid_acp["pred_"+name_dt[i]+"_"+kernel]=pred
    dict_graph_grid_acp["missclass_"+name_dt[i]+"_"+kernel]=missclass
    dict_graph_grid_acp["ecart_"+name_dt[i]+"_"+kernel]=ecart    
    
    
dictionnaire = {"parametres" : hyperparam_choice, "methode_reduction" : type_reduce, "kernel" : kern, 
                "accuracy_train" : accuracy_train, "accuracy_test" : accuracy_test,
                "duree" : duree}
svc_score_grid_acp = pd.DataFrame(dictionnaire)
#svc_graph_grid_acp = pd.DataFrame(dict_graph_grid_acp)    
    
    
#Class prediction
plt.figure(figsize=(22, 12))        
plt.subplot(221)
plt.title("Class prediction : ACP linear")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph_grid_acp["pred_ACP_linear"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)

plt.subplot(222)
plt.title("Class prediction : NMF linear")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph_grid_acp["pred_NMF_linear"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)

plt.subplot(223)
plt.title("Class prediction : No Reduction linear")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph_grid_acp["pred_No Reduction_linear"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements) 
    
    
    
#Missclassified   
plt.figure(figsize=(22, 12))        
plt.subplot(221)
plt.title("Missclassified : ACP linear")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph_grid_acp["missclass_ACP_linear"], s=10, cmap='viridis_r');


plt.subplot(222)
plt.title("Missclassified : NMF linear")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph_grid_acp["missclass_NMF_linear"], s=10, cmap='viridis_r');


plt.subplot(223)
plt.title("Missclassified : No Reduction linear")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph_grid_acp["missclass_No Reduction_linear"], s=10, cmap='viridis_r');

    
    
#Ecart de probabilité

plt.figure(figsize=(22, 12))        
plt.subplot(221)
plt.title("Ecart de probabilité : ACP linear")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph_grid_acp["ecart_ACP_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(222)
plt.title("Ecart de probabilité : NMF linear")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph_grid_acp["ecart_NMF_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(223)
plt.title("Ecart de probabilité : No Reduction linear")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph_grid_acp["ecart_No Reduction_linear"], s=10, cmap='viridis_r');
cbar = plt.colorbar()
       
    
    
    
#Best estimator NMF   
accuracy_train = []
accuracy_test =[]
kern=[]
type_reduce=[]
duree=[]
hyperparam=[]
dict_graph_grid_nmf={}

hyperparam_choice = "Best estimator NMF"
kernel = 'rbf'
for i in range(3): 
    debut = time.time()
    svc_fit=nmf_estimator_grid.fit(x_train[i], y_train[i])
    fin = time.time()
    temps = fin - debut
    pred_train=svc_fit.predict(x_train[i])
    pred_test=svc_fit.predict(x_test[i])
    pred=svc_fit.predict(x[i])
    score_train = metrics.accuracy_score(y_train[i], pred_train)
    score_test = metrics.accuracy_score(y_test[i], pred_test)
    accuracy_train.append(score_train)
    accuracy_test.append(score_test)
    kern.append(ker)
    type_reduce.append(name_dt[i])
    hyperparam.append(hyperparam_choice)
    duree.append(temps)
    probas=svc_fit.predict_proba(x[i])
    ecart=abs(probas[:,0]-probas[:,1])
    missclass = abs(pred-y[i])
    dict_graph_grid_nmf["pred_"+name_dt[i]+"_"+kernel]=pred
    dict_graph_grid_nmf["missclass_"+name_dt[i]+"_"+kernel]=missclass
    dict_graph_grid_nmf["ecart_"+name_dt[i]+"_"+kernel]=ecart    
    
    
dictionnaire = {"parametres" : hyperparam_choice, "methode_reduction" : type_reduce, "kernel" : kern, 
                "accuracy_train" : accuracy_train, "accuracy_test" : accuracy_test,
                "duree" : duree}
svc_score_grid_nmf = pd.DataFrame(dictionnaire)
#svc_graph_grid_acp = pd.DataFrame(dict_graph_grid_acp)    
    
    
#Class prediction
plt.figure(figsize=(22, 12))        

plt.subplot(221)
plt.title("Class prediction : ACP rbf")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph_grid_nmf["pred_ACP_rbf"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)

plt.subplot(222)
plt.title("Class prediction : NMF rbf")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph_grid_nmf["pred_NMF_rbf"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)

plt.subplot(223)
plt.title("Class prediction : No Reduction rbf")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph_grid_nmf["pred_No Reduction_rbf"], s=10, cmap='coolwarm');
plt.legend(handles=legend_elements)    
    
    
    
#Missclassified   
plt.figure(figsize=(22, 12))        

plt.subplot(221)
plt.title("Missclassified : ACP rbf")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph_grid_nmf["missclass_ACP_rbf"], s=10, cmap='viridis_r');


plt.subplot(222)
plt.title("Missclassified : NMF rbf")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph_grid_nmf["missclass_NMF_rbf"], s=10, cmap='viridis_r');


plt.subplot(223)
plt.title("Missclassified : No Reduction rbf")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph_grid_nmf["missclass_No Reduction_rbf"], s=10, cmap='viridis_r');
   
    
    
#Ecart de probabilité

plt.figure(figsize=(22, 12))        

plt.subplot(221)
plt.title("Ecart de probabilité : ACP rbf")
plt.scatter(x[0].iloc[:, 0], x[0].iloc[:, 1], c=dict_graph_grid_nmf["ecart_ACP_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(222)
plt.title("Ecart de probabilité : NMF rbf")
plt.scatter(x[1].iloc[:, 0], x[1].iloc[:, 1], c=dict_graph_grid_nmf["ecart_NMF_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()

plt.subplot(223)
plt.title("Ecart de probabilité : No Reduction rbf")
plt.scatter(x[2].iloc[:, 0], x[2].iloc[:, 1], c=dict_graph_grid_nmf["ecart_No Reduction_rbf"], s=10, cmap='viridis_r');
cbar = plt.colorbar()   
    


svc_score_tout = pd.concat([svc_score, svc_score_grid_acp, svc_score_grid_nmf], axis=0, 
                           ignore_index=True)    
    
svc_score_tout.to_csv(results_path+r'summary_accuracy_SVC.csv', index = False)    
    
    
    
    
    
    
    