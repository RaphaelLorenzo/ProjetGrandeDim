# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:57:18 2021

@author: rapha
"""
#%% Importations
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, validation_curve 
from sklearn.ensemble import BaggingClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option ('display.max_row', 100)
pd.set_option ('display.max_column', 100)


project_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\"
data_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\"

train_a = pd.read_csv(data_path+r"EEG\bsi_competition_ii_train1a.csv", sep = ',')

acp_x_train1a=pd.read_csv(data_path+r"acp_x_train1a.csv")
acp_x_test1a=pd.read_csv(data_path+r"acp_x_test1a.csv")

#8 dimensions
nmf_x_train1a=pd.read_csv(data_path+r"nmf_x_train1a.csv")
nmf_x_test1a=pd.read_csv(data_path+r"nmf_x_test1a.csv")

#5376 dimensions
x_train1a=pd.read_csv(data_path+r"x_train1a.csv")
x_test1a=pd.read_csv(data_path+r"x_test1a.csv")

#y
y_train1a=pd.read_csv(data_path+r"y_train1a.csv")
y_train1a=np.array(y_train1a)[:,0] #sometimes avoids warnings

y_test1a=pd.read_csv(data_path+r"y_test1a.csv")

#Results DataFrame
results_pred=pd.read_csv(project_path+r"Models_Test_Results.csv")
results_pred.columns

#%% Random Forest Classifier
forest = RandomForestClassifier(random_state = 0)
forest.fit(x_train1a, y_train1a)

print("Training set score : {}".format(forest.score(x_train1a, y_train1a)))
print("Test set score : {}".format(forest.score(x_test1a, y_test1a))) #0.776
#Scored by (mean) accuracy

forest.fit(acp_x_train1a, y_train1a)
print("Training set score avec ACP : {}".format(forest.score(acp_x_train1a, y_train1a)))
print("Test set score avec ACP : {}".format(forest.score(acp_x_test1a, y_test1a))) #0.8507

forest.fit(nmf_x_train1a, y_train1a)
print("Training set score avec NMF : {}".format(forest.score(nmf_x_train1a, y_train1a)))
print("Test set score avec NMF : {}".format(forest.score(nmf_x_test1a, y_test1a))) #0.821
 
#%% Optimizing RF Classifier
#ACP
max_depth =np.arange(1, 5, 1)
max_features = ["auto", "log2", "sqrt"]
n_estimators= np.arange (10, 200, 10)

hyper_params = {'max_depth' : max_depth,
               'max_features' : max_features, 
               'n_estimators' : n_estimators}

grid = GridSearchCV (forest, hyper_params, scoring = 'accuracy', cv=5,verbose=1,n_jobs=-1)

grid.fit(acp_x_train1a, y_train1a)

print(grid.best_params_) #{'max_depth': 3, 'max_features': 'auto', 'n_estimators': 90}

y_pred = grid.predict(acp_x_test1a)
results_pred["RF_ACP_Best"]=y_pred

print("Training set score avec ACP : {}".format(grid.score(acp_x_train1a, y_train1a)))
print("Test set score avec ACP : {}".format(grid.score(acp_x_test1a, y_test1a))) #0.8208

N, train_score, val_score = learning_curve(RandomForestClassifier(random_state = 0, max_depth = 3,max_features = 'auto', n_estimators = 90),acp_x_train1a, y_train1a, train_sizes = np.linspace(0.1, 1, 10), cv = 5 )
print(N)

_ = plt.plot(N, train_score.mean(axis = 1), label = 'train')
_ = plt.plot(N, val_score.mean(axis = 1), label = 'validation')
_ = plt.xlabel('train_sizes')
_ = plt.legend()
_ = plt.title("Learning curve for optimal RF Classifier with PCA Datas")



#NMF
grid = GridSearchCV (forest, hyper_params, scoring = 'accuracy', cv=5,verbose=1,n_jobs=-1)

grid.fit(nmf_x_train1a, y_train1a)

print(grid.best_params_)  #{'max_depth': 4, 'max_features': 'auto', 'n_estimators': 20}

y_pred = grid.predict(nmf_x_test1a)
results_pred["RF_NMF_Best"]=y_pred

print("Training set score avec NMF : {}".format(grid.score(nmf_x_train1a, y_train1a)))
print("Test set score avec NMF : {}".format(grid.score(nmf_x_test1a, y_test1a))) #0.8208955223880597

N, train_score, val_score = learning_curve(RandomForestClassifier(random_state = 0, max_depth = 4,max_features = 'auto', n_estimators = 20),nmf_x_train1a, y_train1a, train_sizes = np.linspace(0.1, 1, 10), cv = 5 )
print(N)

_ = plt.plot(N, train_score.mean(axis = 1), label = 'train')
_ = plt.plot(N, val_score.mean(axis = 1), label = 'validation')
_ = plt.xlabel('train_sizes')
_ = plt.legend()
_ = plt.title("Learning curve for optimal RF Classifier with NMF Datas")



#Full Dataset
grid = GridSearchCV (forest, hyper_params, scoring = 'accuracy', cv=5,verbose=1, n_jobs=-1)

grid.fit(x_train1a, y_train1a)

print(grid.best_params_) #{'max_depth': 4, 'max_features': 'log2', 'n_estimators': 30}

y_pred = grid.predict(x_test1a)
results_pred["RF_Full_Best"]=y_pred

print("Training set score avec Full Dataset : {}".format(grid.score(x_train1a, y_train1a)))
print("Test set score avec Full Dataset : {}".format(grid.score(x_test1a, y_test1a))) #0.8059701492537313

N, train_score, val_score = learning_curve(RandomForestClassifier(random_state = 0, max_depth = 4,max_features = 'log2', n_estimators = 30),
                            x_train1a, y_train1a, train_sizes = np.linspace(0.1, 1, 10), cv = 5 )
print(N)

_ = plt.plot(N, train_score.mean(axis = 1), label = 'train')
_ = plt.plot(N, val_score.mean(axis = 1), label = 'validation')
_ = plt.xlabel('train_sizes')
_ = plt.legend()
_ = plt.title("Learning curve for optimal RF Classifier with Full Datas")

#%% Bagging classifier
model = BaggingClassifier(base_estimator = KNeighborsClassifier(), n_estimators = 100,random_state=1)
#NB: Bagging based on KN Classifier

model.fit(x_train1a, y_train1a)
print("Training set score : {}".format(model.score(x_train1a, y_train1a)))
print("Test set score : {}".format(model.score(x_test1a, y_test1a))) #0.835820895522388

model.fit(acp_x_train1a, y_train1a)
print("Training set score avec ACP : {}".format(model.score(acp_x_train1a, y_train1a)))
print("Test set score avec ACP : {}".format(model.score(acp_x_test1a, y_test1a))) #0.835820895522388

model.fit(nmf_x_train1a, y_train1a)
print("Training set score avec NMF : {}".format(model.score(nmf_x_train1a, y_train1a)))
print("Test set score avec NMF : {}".format(model.score(nmf_x_test1a, y_test1a))) #0.7910447761194029


#%% Bagging Optimization
base_estimator=[KNeighborsClassifier (n_neighbors = 2),  KNeighborsClassifier (n_neighbors = 3),  KNeighborsClassifier (n_neighbors = 4),  KNeighborsClassifier (n_neighbors = 5),  KNeighborsClassifier (n_neighbors = 6) ]
max_features = np.arange(1, 10, 1)

hyper_params = {'base_estimator' : base_estimator,
               'max_features' : max_features}

#Full Dataset
grid = GridSearchCV (model, hyper_params, scoring = 'accuracy', cv=5,verbose=1,n_jobs=-1)

grid.fit(x_train1a, y_train1a)

print(grid.best_params_) #{'base_estimator': KNeighborsClassifier(n_neighbors=2), 'max_features': 7}

y_pred = grid.predict(x_test1a)
results_pred["BaggingKN_Full_Best"]=y_pred

print("Training set score avec Full Dataset : {}".format(grid.score(x_train1a, y_train1a)))
print("Test set score avec Full Dataset : {}".format(grid.score(x_test1a, y_test1a))) #0.8507462686567164


#ACP
grid = GridSearchCV (model, hyper_params, scoring = 'accuracy', cv=5,verbose=1,n_jobs=-1)

grid.fit(acp_x_train1a, y_train1a) 

print(grid.best_params_) #{'base_estimator': KNeighborsClassifier(n_neighbors=4), 'max_features': 5}

y_pred = grid.predict(acp_x_test1a)
results_pred["BaggingKN_ACP_Best"]=y_pred

print("Training set score avec ACP : {}".format(grid.score(acp_x_train1a, y_train1a)))
print("Test set score avec ACP : {}".format(grid.score(acp_x_test1a, y_test1a))) #0.8656716417910447

#NMF
grid = GridSearchCV (model, hyper_params, scoring = 'accuracy', cv=5,verbose=1,n_jobs=-1)

grid.fit(nmf_x_train1a, y_train1a)

print(grid.best_params_) #{'base_estimator': KNeighborsClassifier(n_neighbors=3), 'max_features': 3}

y_pred = grid.predict(nmf_x_test1a)
results_pred["BaggingKN_NMF_Best"]=y_pred

print("Training set score avec NMF : {}".format(grid.score(nmf_x_train1a, y_train1a)))
print("Test set score avec NMF : {}".format(grid.score(nmf_x_test1a, y_test1a))) #0.8208955223880597


#%% AdaBoost

#Add time comparision between SAMME and SAMME.R

adaboost = AdaBoostClassifier(random_state = 0)

#Full Dataset
adaboost.fit(x_train1a, y_train1a)
print("Training set score : {}".format(adaboost.score(x_train1a, y_train1a)))
print("Test set score : {}".format(adaboost.score(x_test1a, y_test1a))) #0.835820895522388

#ACP
adaboost.fit(acp_x_train1a, y_train1a)

print("Training set score avec ACP : {}".format(adaboost.score(acp_x_train1a, y_train1a)))
print("Test set score avec ACP : {}".format(adaboost.score(acp_x_test1a, y_test1a))) #0.8208955223880597

#NMF
adaboost.fit(nmf_x_train1a, y_train1a)
print("Training set score avec ACP : {}".format(adaboost.score(nmf_x_train1a, y_train1a)))
print("Test set score avec ACP : {}".format(adaboost.score(nmf_x_test1a, y_test1a))) #0.8507462686567164


#%% Optimizing AdaBoost
adaboost = AdaBoostClassifier(random_state = 0)

# learning_rate=np.arange(0.2, 1.5, 0.2)
# n_estimators= np.arange (10, 200, 25)
base_estimator=[]

for k in range(1, 6, 1) :
    base_estimator=base_estimator+[DecisionTreeClassifier(max_depth=k)] 

# hyper_params = {'learning_rate' : learning_rate,
#                'n_estimators' : n_estimators,
#                'base_estimator' : base_estimator}
#Simply hyper params to get faster CV

hyper_params = {'base_estimator' : base_estimator}

grid = GridSearchCV (adaboost, hyper_params, scoring = 'accuracy', cv=5,verbose=1,n_jobs=-1)

#Full Dataset
grid.fit(x_train1a, y_train1a)

print(grid.best_params_) #{'base_estimator': DecisionTreeClassifier(max_depth=4)}

y_pred = grid.predict(x_test1a)
results_pred["AdaBoostDT_Full_Best"]=y_pred

print("Training set score avec Full Dataset : {}".format(grid.score(x_train1a, y_train1a)))
print("Test set score avec Full Dataset : {}".format(grid.score(x_test1a, y_test1a))) #0.835820895522388

#ACP
learning_rate=np.arange(0.2, 1.5, 0.2)
n_estimators= np.arange (10, 200, 25)
hyper_params = {'learning_rate' : learning_rate,'n_estimators' : n_estimators,'base_estimator' : base_estimator}
grid = GridSearchCV (adaboost, hyper_params, scoring = 'accuracy', cv=5,verbose=1,n_jobs=-1)

grid.fit(acp_x_train1a, y_train1a) 

print(grid.best_params_)  #{'base_estimator': DecisionTreeClassifier(max_depth=1), 'learning_rate': 1.2, 'n_estimators': 135}

y_pred = grid.predict(acp_x_test1a)
results_pred["AdaBoostDT_ACP_Best"]=y_pred

print("Training set score avec ACP : {}".format(grid.score(acp_x_train1a, y_train1a)))
print("Test set score avec ACP : {}".format(grid.score(acp_x_test1a, y_test1a))) #0.8208955223880597

#NMF
grid = GridSearchCV (adaboost, hyper_params, scoring = 'accuracy', cv=5,verbose=1,n_jobs=-1)

grid.fit(nmf_x_train1a, y_train1a)

print(grid.best_params_) #{'base_estimator': DecisionTreeClassifier(max_depth=3), 'learning_rate': 0.8, 'n_estimators': 160}

y_pred = grid.predict(nmf_x_test1a)
results_pred["AdaBoostDT_NMF_Best"]=y_pred

print("Training set score avec NMF : {}".format(grid.score(nmf_x_train1a, y_train1a)))
print("Test set score avec NMF : {}".format(grid.score(nmf_x_test1a, y_test1a))) #0.7761194029850746


#%%
results_pred.to_csv(project_path+r"Models_Test_Results.csv",index=False)
