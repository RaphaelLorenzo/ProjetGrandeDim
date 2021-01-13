# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:02:40 2021

@author: shade
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import math
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

pd.set_option ('display.max_row', 12)
pd.set_option ('display.max_column', 12)
plt.style.use('seaborn-darkgrid')

# traina = pd.read_csv(r'C:\Users\shade\OneDrive\Documents\M2_TIDE\Analyse_grande_dimension\Projet\EEG Dataset-20201226\EEG_Full.csv\bsi_competition_ii_train1a.csv')
# testa = pd.read_csv(r'C:\Users\shade\OneDrive\Documents\M2_TIDE\Analyse_grande_dimension\Projet\EEG Dataset-20201226\EEG_Full.csv\bsi_competition_ii_test1a.csv')


# traina.drop('0', axis=1, inplace=True)

# scaler = StandardScaler()
# traina_scaled = scaler.fit_transform(traina)
# testa_scaled = scaler.fit_transform(testa)
# traina_scaled = pd.DataFrame(data = traina_scaled)
# testa_scaled = pd.DataFrame(data = testa_scaled)

traina_scaled=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\scaled_train_1a.csv")
testa_scaled=pd.read_csv(r"C:\Users\rapha\Desktop\ProjetGrandeDIm_Local\scaled_test_1a.csv")
y_traina=traina_scaled["0"]
traina_scaled=traina_scaled.drop("0",axis=1)

duree_full=[]
duree_arpack=[]
duree_random=[]

svd_solver=["full", "arpack", "randomized"]
for n in range(1, 20):
    for solver in svd_solver :
        acp = PCA(n_components=n, svd_solver=solver)
        debut = time.time()
        acp.fit(traina_scaled)
        fin = time.time()
        duree = fin - debut
        print(solver + ' : '+ str(duree)+' seconds for '+str(n)+' components')
        if solver=="full":
            duree_full.append(duree)
        elif solver=="arpack":
            duree_arpack.append(duree)
        else:
            duree_random.append(duree)
    print("***************")


#Plot durée
plt.figure()
plt.plot(range(1, 20), duree_full, color = 'red',label="Durée full")
plt.plot(range(1, 20), duree_arpack, color = 'blue',label="Durée arpack")
plt.plot(range(1, 20), duree_random, color = 'black',label="Durée randomized")
plt.legend()
plt.title('Durée');

#Projection de test
acp_arpack=PCA(n_components=19, svd_solver='arpack')
traina_arpack=acp_arpack.fit_transform(traina_scaled)

exp_vr_ratio = acp_arpack.explained_variance_ratio_
exp_vr_ratio = pd.DataFrame(data = exp_vr_ratio)

vr_ratio_sum = exp_vr_ratio.cumsum()

#traina_arpack = pd.DataFrame(data = traina_arpack)


plt.figure()
plt.plot(range(1, 20), vr_ratio_sum, color = 'red')
plt.title('Explained variance ratio');

lr_score = []
for i in range(1, 20):   
    lr_cv = LogisticRegressionCV(cv = 2, multi_class = "multinomial")
    lrcv_fit = lr_cv.fit(traina_arpack[:, 0:i], y_traina)
    pred_train = lrcv_fit.predict(traina_arpack[:, 0:i])
    score_train = metrics.accuracy_score(y_traina, pred_train)
    lr_score.append(score_train)


plt.style.use('default')
fig, ax1 = plt.subplots()

ax1.set_xlabel('Components')
ax1.set_ylabel('Accuracy of Logistic Regression',color="blue")
ax1.plot(range(1,20),lr_score,color="blue")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  

ax2.set_ylabel("Explained variance ratio",color="red")  
ax2.plot(range(1,20),vr_ratio_sum,color="red")
ax2.tick_params(axis='y')

fig.tight_layout()


df_traina_arpack=pd.DataFrame(data=traina_arpack[:,0:9], columns=["Dim"+str(i) for i in range(1,10)])
df_traina_arpack["label"]=y_traina


testa_arpack= acp_arpack.transform(testa_scaled)
df_testa_arpack = pd.DataFrame(data = testa_arpack[:,0:9], columns=["Dim"+str(i) for i in range(1,10)])


df_traina_arpack.to_csv(r'C:\Users\shade\OneDrive\Documents\M2_TIDE\Analyse_grande_dimension\Projet\EEG Dataset-20201226\EEG_Full.csv\acp_train1a.csv', sep = ",", index = False)

df_testa_arpack.to_csv(r'C:\Users\shade\OneDrive\Documents\M2_TIDE\Analyse_grande_dimension\Projet\EEG Dataset-20201226\EEG_Full.csv\acp_test1a.csv', sep = ",", index = False)

acp_traina = pd.read_csv(r'C:\Users\shade\OneDrive\Documents\M2_TIDE\Analyse_grande_dimension\Projet\EEG Dataset-20201226\EEG_Full.csv\acp_train1a.csv')

acp_testa = pd.read_csv(r'C:\Users\shade\OneDrive\Documents\M2_TIDE\Analyse_grande_dimension\Projet\EEG Dataset-20201226\EEG_Full.csv\acp_test1a.csv')










