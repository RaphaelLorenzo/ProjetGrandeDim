# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:24:04 2020

@author: rapha
"""
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import cm

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
import time

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from  sklearn import metrics 
plt.style.use('seaborn-darkgrid')


project_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\"
data_path=r"C:\\Users\\rapha\\Desktop\\TIDE S1\\ProjetGrandeDIm_Local\\"


scaled_test_1a=pd.read_csv(project_path+r"scaled_test_1a.csv")
scaled_train_1a=pd.read_csv(project_path+r"scaled_train_1a.csv")

#Non negative matrix for NMF, methods nneg with min
scaled_train_1a.iloc[:,1:]=scaled_train_1a.iloc[:,1:]+abs(scaled_train_1a.iloc[:,1:].min().min())
scaled_train_1a.columns=range(5377)

scaled_test_1a=scaled_test_1a+abs(scaled_test_1a.min().min())

#We will be checking 1 to 50 components

#%% All default NMF : nmf_1
fit_time=[]
error_rate=[]

for n_comp in range(1,20):
    nmf_1=decomposition.NMF(n_components=n_comp)
    time_start=time.time()
    nmf_1.fit(scaled_train_1a.iloc[:,1:])
    time_stop=time.time()
    
    timelength=time_stop-time_start
    print("NMF with default parameters fit X in "+str(timelength)+" seconds with "+str(n_comp)+" components")
    
    error=nmf_1.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro") #Error of the NMF

    fit_time.append(timelength)
    error_rate.append(error)

plt.figure()
plt.plot(range(1,20),fit_time)
plt.xlabel("Components")
plt.ylabel("Time of fit in seconds")
plt.title("Time of fit for default NMF")

plt.figure()
plt.plot(range(1,20),error_rate)
plt.xlabel("Components")
plt.ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$")
plt.title("Error rate for default NMF")

#2D data visualization
nmf_1=decomposition.NMF(n_components=2)
nmf_1.fit(scaled_train_1a.iloc[:,1:])

nmf_scaled_train1a=nmf_1.transform(scaled_train_1a.iloc[:,1:])

plt.scatter(nmf_scaled_train1a[:,0],nmf_scaled_train1a[:,1],c=scaled_train_1a.iloc[:,0],cmap="viridis")
plt.title("NMF Decomposition (2 components)")

#%%Increase max Iterations as suggested : nmf_2
fit_time=[]
error_rate=[]

#Warning : LONG TO EXECUTE (approx 5 minutes)
#With fix components number at 6
for max_iter in range(1000,6000,500):
    nmf_2=decomposition.NMF(n_components=6,max_iter=max_iter)
    time_start=time.time()
    nmf_2.fit(scaled_train_1a.iloc[:,1:])
    time_stop=time.time()
    
    timelength=time_stop-time_start
    print("NMF fit X in "+str(timelength)+" seconds with "+str(max_iter)+" max iterations")
    
    error=nmf_2.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro") #Error of the NMF

    fit_time.append(timelength)
    error_rate.append(error)

nmf_2.n_iter_

plt.figure()
plt.plot(range(1000,6000,500),fit_time)
plt.xlabel("Iterations max")
plt.ylabel("Time of fit in seconds")
plt.title("Time of fit for NMF")

plt.figure()
plt.plot(range(1000,6000,500),error_rate)
plt.xlabel("Iterations max")
plt.ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$")
plt.title("Error rate for NMF")

#4698 iterations and a long time required but a tiny improvement : keep it limited by setting the 
#max_iter to 1000 and the tol to 0.001 instead of 0.0001
#leads to 480 iterations in our present setting


#%% Check various initialization methods : nmf_3 with nmf_3_nndsvd and nmf_3_random 
init=["nndsvd","random"]

fit_time_nndsvd=[]
error_rate_nndsvd=[]
n_iter_nndsvd=[]

fit_time_random=[]
error_rate_random=[]
n_iter_random=[]

for n_comp in range(1,10,1):
    nmf_3_nndsvd=decomposition.NMF(n_components=n_comp,max_iter=1000,tol=0.001,init="nndsvd")
    time_start=time.time()
    nmf_3_nndsvd.fit(scaled_train_1a.iloc[:,1:])
    time_stop=time.time()
    
    timelength=time_stop-time_start
    
    print("NMF with nndsvd initialization fit X in "+str(timelength)+" seconds with "+ str(n_comp) +" components and "+str(nmf_3_nndsvd.n_iter_)+" actual iterations")
    
    error=nmf_3_nndsvd.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro") #Error of the NMF

    fit_time_nndsvd.append(timelength)
    error_rate_nndsvd.append(error)
    n_iter_nndsvd.append(nmf_3_nndsvd.n_iter_)
    
    nmf_3_random=decomposition.NMF(n_components=n_comp,max_iter=1000,tol=0.001,init="random")
    time_start=time.time()
    nmf_3_random.fit(scaled_train_1a.iloc[:,1:])
    time_stop=time.time()
    
    timelength=time_stop-time_start
    
    print("NMF with random initialization fit X in "+str(timelength)+" seconds with "+ str(n_comp) +" components and "+str(nmf_3_random.n_iter_)+" actual iterations")
    
    error=nmf_3_random.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro") #Error of the NMF

    fit_time_random.append(timelength)
    error_rate_random.append(error)   
    n_iter_random.append(nmf_3_random.n_iter_)


plt.figure()
plt.plot(range(1,10,1),fit_time_nndsvd,label="NNDSVD")
plt.plot(range(1,10,1),fit_time_random,label="random")
plt.xlabel("Components")
plt.ylabel("Time of fit in seconds")
plt.legend()
plt.title("Time of fit for NMF with various initializations")

plt.figure()
plt.plot(range(1,10,1),n_iter_nndsvd,label="NNDSVD")
plt.plot(range(1,10,1),n_iter_random,label="random")
plt.xlabel("Components")
plt.ylabel("Iterations")
plt.legend()
plt.title("Number of iterations required for NMF with various initializations")


plt.figure()
plt.plot(range(1,10,1),error_rate_nndsvd,label="NNDSVD")
plt.plot(range(1,10,1),error_rate_random,label="random")
plt.xlabel("Components")
plt.ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$")
plt.legend()
plt.title("Error rate for NMF with various initializations")

#As expected almost exactly the same error level.
#The random init seems slightly more efficient in terms of time



#%% Compare solver algorithm : nmf_4 with nmf_4_mu and nmf_4_cd
#MU algorithm always with beta_loss="froebenius" (default)
#keep a random init but with a fixed random_state so that things keep comparable

fit_time_mu=[]
error_rate_mu=[]
n_iter_mu=[]

fit_time_cd=[]
error_rate_cd=[]
n_iter_cd=[]

for n_comp in range(1,20,1):
    nmf_4_mu=decomposition.NMF(n_components=n_comp,max_iter=1000,tol=0.001,init="random",solver="mu",random_state=1)
    time_start=time.time()
    nmf_4_mu.fit(scaled_train_1a.iloc[:,1:])
    time_stop=time.time()
    
    timelength=time_stop-time_start
    
    print("NMF with random initialization and MU fit X in "+str(timelength)+" seconds with "+ str(n_comp) +" components and "+str(nmf_4_mu.n_iter_)+" actual iterations")
    
    error=nmf_4_mu.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro") #Error of the NMF

    fit_time_mu.append(timelength)
    error_rate_mu.append(error)
    n_iter_mu.append(nmf_4_mu.n_iter_)
    
    nmf_4_cd=decomposition.NMF(n_components=n_comp,max_iter=1000,tol=0.001,init="random",random_state=1,solver="cd")
    time_start=time.time()
    nmf_4_cd.fit(scaled_train_1a.iloc[:,1:])
    time_stop=time.time()
    
    timelength=time_stop-time_start
    
    print("NMF with random initialization and CD fit X in "+str(timelength)+" seconds with "+ str(n_comp) +" components and "+str(nmf_4_cd.n_iter_)+" actual iterations")
    
    error=nmf_4_cd.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro") #Error of the NMF

    fit_time_cd.append(timelength)
    error_rate_cd.append(error)   
    n_iter_cd.append(nmf_4_cd.n_iter_)


plt.figure()
plt.plot(range(1,20,1),fit_time_mu,label="Multiplicative Update")
plt.plot(range(1,20,1),fit_time_cd,label="Alternate Coordinate Descent")
plt.xlabel("Components")
plt.ylabel("Time of fit in seconds")
plt.legend()
plt.title("Time of fit for NMF with various algorithm")

plt.figure()
plt.plot(range(1,20,1),n_iter_mu,label="Multiplicative Update")
plt.plot(range(1,20,1),n_iter_cd,label="Alternate Coordinate Descent")
plt.xlabel("Components")
plt.ylabel("Iterations")
plt.legend()
plt.title("Number of iterations required for NMF with various algorithm")


plt.figure()
plt.plot(range(1,20,1),error_rate_mu,label="Multiplicative Update")
plt.plot(range(1,20,1),error_rate_cd,label="Alternate Coordinate Descent")
plt.xlabel("Components")
plt.ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$")
plt.legend()
plt.title("Error rate for NMF with various algorithm")

#Multiplicative update is faster for high number of components but for n_components high the result error is slightly higher 

#2D visualization
nmf_4_mu=decomposition.NMF(n_components=2,max_iter=1000,tol=0.001,init="random",solver="mu",random_state=1)
nmf_4_mu.fit(scaled_train_1a.iloc[:,1:])
nmf_scaled_train1a=nmf_4_mu.transform(scaled_train_1a.iloc[:,1:])

plt.figure()
plt.scatter(nmf_scaled_train1a[:,0],nmf_scaled_train1a[:,1],c=scaled_train_1a.iloc[:,0],cmap="viridis")
plt.title("NMF representation with 2 components fitted by MU")


nmf_4_cd=decomposition.NMF(n_components=2,max_iter=1000,tol=0.001,init="random",random_state=1,solver="cd")
nmf_4_cd.fit(scaled_train_1a.iloc[:,1:])
nmf_scaled_train1a=nmf_4_cd.transform(scaled_train_1a.iloc[:,1:])

plt.figure()
plt.scatter(nmf_scaled_train1a[:,0],nmf_scaled_train1a[:,1],c=scaled_train_1a.iloc[:,0],cmap="viridis")
plt.title("NMF representation with 2 components fitted by CD")

nmf_4_cd.reconstruction_err_
nmf_4_mu.reconstruction_err_

#%% Regularization parameters alpha and l1_ratio : nmf_5
#We keep a regularization = "both" (on W and H, or X and V, both components and transformation matrix) or a regularization="transformation" (on the individuals) #not available in our version (only the 0.24.0 - the last one)

#L1 regularization
n_zeros=[]
error_rate=[]
for alpha in range(1,50,1):
    nmf_5=decomposition.NMF(n_components=6,max_iter=1000,tol=0.001,init="nndsvd",solver="cd",random_state=1,alpha=alpha,l1_ratio=1)
    nmf_5.fit(scaled_train_1a.iloc[:,1:])
    
    
    nmf_scaled_train1a=nmf_5.transform(scaled_train_1a.iloc[:,1:])
    #plt.figure()
    #plt.scatter(nmf_scaled_train1a[:,0],nmf_scaled_train1a[:,1],c=scaled_train_1a.iloc[:,0],cmap="viridis")
    #plt.title("2D representation with alpha = "+str(alpha))
    spars=(nmf_scaled_train1a==0).sum(axis=0)
    print("alpha = "+str(alpha))
    print(spars)
    n_zeros.append(spars.tolist())
    
    error=nmf_5.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro") #Error of the NMF
    error_rate.append(error)

n_zeros=pd.DataFrame(n_zeros)

plt.figure()
plt.plot(range(1,50,1),n_zeros.iloc[:,0],label="Component 1")
plt.plot(range(1,50,1),n_zeros.iloc[:,1],label="Component 2")
plt.plot(range(1,50,1),n_zeros.iloc[:,2],label="Component 3")
plt.plot(range(1,50,1),n_zeros.iloc[:,3],label="Component 4")
plt.plot(range(1,50,1),n_zeros.iloc[:,4],label="Component 5")
plt.plot(range(1,50,1),n_zeros.iloc[:,5],label="Component 6")
plt.plot(range(1,50,1),n_zeros.iloc[:,:].sum(axis=1),label="Total")
plt.title("Sparsity with L1 constraint on NMF matrix")
plt.ylabel("Number of 0 values")
plt.xlabel(r"Value of $\alpha$")
plt.legend()

plt.figure()
plt.plot(range(1,50,1),error_rate)
plt.xlabel(r"Value of $\alpha$")
plt.ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$")
plt.title("Error rate for NMF with L1 constraint (6 components)")

#Froebenius regularization
n_zeros=[]
error_rate=[]
for alpha in range(1,50,1):
    nmf_5=decomposition.NMF(n_components=6,max_iter=1000,tol=0.001,init="nndsvd",solver="cd",random_state=1,alpha=alpha,l1_ratio=0)
    nmf_5.fit(scaled_train_1a.iloc[:,1:])
    
    
    nmf_scaled_train1a=nmf_5.transform(scaled_train_1a.iloc[:,1:])
    #plt.figure()
    #plt.scatter(nmf_scaled_train1a[:,0],nmf_scaled_train1a[:,1],c=scaled_train_1a.iloc[:,0],cmap="viridis")
    #plt.title("2D representation with alpha = "+str(alpha))
    spars=(nmf_scaled_train1a==0).sum(axis=0)
    print("alpha = "+str(alpha))
    print(spars)
    n_zeros.append(spars.tolist())
    error=nmf_5.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro") #Error of the NMF
    error_rate.append(error)


n_zeros=pd.DataFrame(n_zeros)

plt.figure()
plt.plot(range(1,50,1),n_zeros.iloc[:,0],label="Component 1")
plt.plot(range(1,50,1),n_zeros.iloc[:,1],label="Component 2")
plt.plot(range(1,50,1),n_zeros.iloc[:,2],label="Component 3")
plt.plot(range(1,50,1),n_zeros.iloc[:,3],label="Component 4")
plt.plot(range(1,50,1),n_zeros.iloc[:,4],label="Component 5")
plt.plot(range(1,50,1),n_zeros.iloc[:,5],label="Component 6")
plt.plot(range(1,50,1),n_zeros.iloc[:,:].sum(axis=1),label="Total")
plt.title("Sparsity with Froebenius constraint on NMF matrix")
plt.ylabel("Number of 0 values")
plt.xlabel(r"Value of $\alpha$")
plt.legend()

plt.figure()
plt.plot(range(1,50,1),error_rate)
plt.xlabel(r"Value of $\alpha$")
plt.ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$")
plt.title("Error rate for NMF with Froebenius constraint (6 components)")


#Use a manual CV grid to gest the best couple of parameters to predict, with 6 components


l1_ratio_range =np.arange(0,1.2, 0.2)
alpha_range = np.arange(0, 6, 1)

X,Y=np.meshgrid(alpha_range,l1_ratio_range)

Z=np.array(Y)
T=np.array(Y)
I=np.array(Y)

start=time.time()
for i in range(Z.shape[0]): 
    #sur la ième lignes
    #print("alpha="+str(Y[i,0]))
    for j in range(Z.shape[1]):
        #print("l1_ratio="+str(X[0,j]))
        #print("alpha="+str(Y[i,0]))
        alpha=Y[i,0]
        l1_ratio=X[0,j]
        nmf_5=decomposition.NMF(n_components=6,max_iter=1000,tol=0.01,init="nndsvd",solver="cd",random_state=1,alpha=alpha,l1_ratio=l1_ratio)
        #tol raised to avoid too much warnings and a too long NMF
        nmf_5.fit(scaled_train_1a.iloc[:,1:])
        Z[i,j]=nmf_5.reconstruction_err_
        nmf_scaled_train1a=nmf_5.transform(scaled_train_1a.iloc[:,1:])
        
        y_true=scaled_train_1a.iloc[:,0]
        
        lr=LogisticRegression()
        lr=lr.fit(nmf_scaled_train1a,y_true)
        y_pred=lr.predict(nmf_scaled_train1a)
        
        lr_acc = metrics.accuracy_score(y_pred, y_true)
        print(lr_acc)
        T[i,j]=lr_acc
        I[i,j]=nmf_5.reconstruction_err_.n_iter_
        
end=time.time()
print("Done in : "+str(end-start)+" seconds")


#Using parallelization
#Efficient but for simpler processes, causes error (infinite values, full 0 matrix... that does not occur with standard processing)
#So some lines a "manually" computed
#Eg : ValueError: Array passed to NMF (input H) is full of zeros.
#Eg : ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

# l1_ratio_range =np.arange(0,1.05, 0.05)
# alpha_range = np.arange(0, 51, 1)

# X,Y=np.meshgrid(alpha_range,l1_ratio_range)

# Z=np.array(Y)
# T=np.array(Y)
# I=np.array(Y)
# def myfun(i,j):
#     alpha=Y[i,0]
#     l1_ratio=X[0,j]
#     nmf_5=decomposition.NMF(n_components=6,max_iter=1000,tol=0.01,init="nndsvd",solver="cd",random_state=1,alpha=alpha,l1_ratio=l1_ratio)
#     #tol raised to avoid too much warnings and a too long NMF
#     nmf_5.fit(scaled_train_1a.iloc[:,1:])
#     #Z[i,j]=nmf_5.reconstruction_err_
#     nmf_scaled_train1a=nmf_5.transform(scaled_train_1a.iloc[:,1:])
    
#     y_true=scaled_train_1a.iloc[:,0]
    
#     lr=LogisticRegression()
#     lr=lr.fit(nmf_scaled_train1a,y_true)
#     y_pred=lr.predict(nmf_scaled_train1a)
    
#     lr_acc = metrics.accuracy_score(y_pred, y_true)
#     #T[i,j]=lr_acc     
#     #I[i,j]=nmf_5.reconstruction_err_.n_iter_
#     return nmf_5.reconstruction_err_,lr_acc,nmf_5.n_iter_

 

# start=time.time()
# for i in range(16,21):
#     num_cores = multiprocessing.cpu_count()
#     inputs = tqdm(range(len(alpha_range)))
#     print("line "+str(i))
#     if __name__ == "__main__":
#         processed_list = Parallel(n_jobs=num_cores)(delayed(myfun)(i,k) for k in inputs)
       
#     Z[i,:]=[processed_list[t][0] for t in range(51)]
#     T[i,:]=[processed_list[t][1] for t in range(51)]
#     I[i,:]=[processed_list[t][2] for t in range(51)]

# end=time.time()
# print("Done in : "+str(end-start)+" seconds")

# #Manual line in case of error
# i=15
# processed_list=[]
# for k in range(51):
#     print(k)
#     processed_list.append(myfun(i,k))


# Z[i,:]=[processed_list[t][0] for t in range(51)]
# T[i,:]=[processed_list[t][1] for t in range(51)]
# I[i,:]=[processed_list[t][2] for t in range(51)]



#This procedure is already very long but it could have been more done with more 
#rigorous method by using cross validation and getting the mean of the accuracy 
#on each Fold for each (alpha,l1_ratio)


fig, ax = plt.subplots(figsize=(12, 12),subplot_kw={"projection": "3d"})

ax.plot_surface(X, Y, np.log(Z), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel("Alpha")
ax.set_ylabel("L1 Ratio")
ax.set_zlabel("Log Error")
ax.set_title("Reconstruction error level")

np.unravel_index(Z.argmin(), Z.shape)
print("Best alpha = "+str(alpha_range[np.unravel_index(Z.argmin(), Z.shape)[1]]))
print("Best L1 Ratio = "+str(l1_ratio_range[np.unravel_index(Z.argmin(), Z.shape)[0]]))


fig, ax = plt.subplots(figsize=(12, 12),subplot_kw={"projection": "3d"})

ax.plot_surface(X, Y, T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel("Alpha")
ax.set_ylabel("L1 Ratio")
ax.set_zlabel("Logistic Regression Accuracy")
ax.set_title("Accuracy of the Logistic Regression")

np.unravel_index(T.argmax(), T.shape)

print("Best alpha = "+str(alpha_range[np.unravel_index(T.argmax(), T.shape)[1]]))
print("Best L1 Ratio = "+str(l1_ratio_range[np.unravel_index(T.argmax(), T.shape)[0]]))
print("Best accuracy = "+str(T.max()))

fig, ax = plt.subplots(figsize=(12, 12),subplot_kw={"projection": "3d"})

ax.plot_surface(X, Y, I, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel("Alpha")
ax.set_ylabel("L1 Ratio")
ax.set_zlabel("Iterations")
ax.set_title("Number of iterations") 

#Unfortunately a lot of iterations don't reach their optimum because of a lack of iterations, but the nmf is performed quite fast with low alpha (whatever the L1 ratio is)

#Adding any sort of regularization does not actually improve ou results

#%% Choosing the number of components for a prediction aim : nmf_6

#As seen above the influence of regularization parameters, and the sparsity of the matrix (given it has a relatively small number of components anyway) does not matter in the quality of the Logistic Regression prediction
#It is also making the solving process harder (when alpha is high)

#We shall keep alpha=0, use a random init, and a CD solver
error_rate=[]
lr_result=[]

y=scaled_train_1a.iloc[:,0]
for n_comp in tqdm(range(1,20,1)):
    
    nmf_6=decomposition.NMF(n_components=n_comp,max_iter=1000,tol=0.001,init="random",solver="cd",random_state=1)
    nmf_6.fit(scaled_train_1a.iloc[:,1:])
    
    error=nmf_6.reconstruction_err_/np.linalg.norm(scaled_train_1a.iloc[:,1:],ord="fro")
    
    nmf_scaled_train1a=nmf_6.transform(scaled_train_1a.iloc[:,1:])
    
    kf = KFold(n_splits=2,shuffle=True)
    
    lr_acc_cv=[]
    i=0
    for train_index, test_index in kf.split(nmf_scaled_train1a):
        i+=1
        print("Fold number"+str(i))
        X_train, X_test = nmf_scaled_train1a[train_index], nmf_scaled_train1a[test_index]
        y_train, y_test = y[train_index], y[test_index]
     
        y_true=y_test
        
        lr=LogisticRegression()
        lr=lr.fit(X_train,y_train)
        y_pred=lr.predict(X_test)
        lr_acc_l = metrics.accuracy_score(y_pred, y_true)
        lr_acc_cv.append(lr_acc_l)
    
    error_rate.append(error)
    lr_result.append(np.mean(lr_acc_cv))


        
plt.style.use('default')
fig, ax1 = plt.subplots()

ax1.set_xlabel('Components')
ax1.set_ylabel('Accuracy of Logistic Regression',color="blue")
ax1.plot(range(1,20,1),lr_result,label="Logistic Regression Accuracy",color="blue")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  

ax2.set_ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$",color="red")  
ax2.plot(range(1,20,1),error_rate,label="Reconstruction Error Rate",color="red")
ax2.tick_params(axis='y')
plt.title("Accuracy of LR (mean of 2 folds) \n and Reconstruction Error \n (NMF performed and error measured on the whole dataset \n before splitting into train and test datas) ")

fig.tight_layout()

#%% Let us be more rigorous on the decomposition approaches


def Proj_Type(type_proj,nFold,n_compMax):
    compos=[]
    error_rate_train=[]
    error_rate_test=[]
    lr_result=[]
    y=scaled_train_1a.iloc[:,0]
    for n_comp in tqdm(range(1,n_compMax,1)):
        
        nmf_6=decomposition.NMF(n_components=n_comp,max_iter=1000,tol=0.001,init="random",solver="cd",random_state=1)
        
        kf = KFold(n_splits=nFold,shuffle=True)
        
        lr_acc_cv=[]
        error_rate_train_cv=[]
        error_rate_test_cv=[]
        i=0
        for train_index, test_index in kf.split(scaled_train_1a.iloc[:,1:]):
            i+=1
            print("Fold number"+str(i))
            X_train, X_test = scaled_train_1a.iloc[train_index,1:], scaled_train_1a.iloc[test_index,1:]
            y_train, y_test = y[train_index], y[test_index]
            
            if type_proj=="Own":
                X_train_nmf=nmf_6.fit_transform(X_train)
                error=nmf_6.reconstruction_err_/np.linalg.norm(X_train,ord="fro") #reconstruction error measured on the train data nmf
                error_rate_train_cv.append(error)
                
                X_test_nmf=nmf_6.fit_transform(X_test)
                error=nmf_6.reconstruction_err_/np.linalg.norm(X_test,ord="fro") #reconstruction error measured on the test data nmf
                error_rate_test_cv.append(error)
                
                y_true=y_test
                
                lr=LogisticRegression()
                lr=lr.fit(X_train_nmf,y_train)
                y_pred=lr.predict(X_test_nmf)
                lr_acc_l = metrics.accuracy_score(y_pred, y_true)
                lr_acc_cv.append(lr_acc_l)
                
            elif type_proj=="Train":
                nmf_6.fit(X_train)
                compo=nmf_6.components_
                compos.append(compo)
                X_train_nmf=nmf_6.transform(X_train)
                error=np.linalg.norm(X_train-np.dot(X_train_nmf,compo),ord="fro")/np.linalg.norm(X_train,ord="fro") 
                #error=nmf_6.reconstruction_err_/np.linalg.norm(X_train,ord="fro") #reconstruction error measured on the train data nmf
                error_rate_train_cv.append(error)
    
                X_test_nmf=nmf_6.transform(X_test)
                error=np.linalg.norm(X_test-np.dot(X_test_nmf,compo),ord="fro")/np.linalg.norm(X_test,ord="fro") #reconstruction error measured on the test data nmf
                #must be caculated "manually" because nmf_6 is not fitted on X_test so reconstruction_err_ is the one of X_train NMF
                error_rate_test_cv.append(error)
              
                y_true=y_test
                
                lr=LogisticRegression()
                lr=lr.fit(X_train_nmf,y_train)
                y_pred=lr.predict(X_test_nmf)
                lr_acc_l = metrics.accuracy_score(y_pred, y_true)
                lr_acc_cv.append(lr_acc_l)           
        
        error_rate_train.append(np.mean(error_rate_train_cv))
        error_rate_test.append(np.mean(error_rate_test_cv))
        
        lr_result.append(np.mean(lr_acc_cv))
        
    return error_rate_train, error_rate_test, lr_result, compos


#Compare the results : first each matrix (train and test) get their own NMF with a great reconstruction
error_rate_train_own, error_rate_test_own, lr_result_own, compos = Proj_Type(type_proj="Own",nFold=2,n_compMax=20)


plt.style.use('default')
fig, ax1 = plt.subplots()

ax1.set_xlabel('Components')
ax1.set_ylabel('Accuracy of Logistic Regression',color="blue")
ax1.plot(range(1,20,1),lr_result_own,label="Logistic Regression Accuracy",color="blue")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  

ax2.set_ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$",color="red")  
ax2.plot(range(1,20,1),error_rate_train_own,color="red",label="Train Reconstruction")
ax2.plot(range(1,20,1),error_rate_test_own,color="red",linestyle=':',label="Test Reconstruction")
ax2.legend()
ax2.tick_params(axis='y')
plt.title("Accuracy of LR (mean of 2 folds) \n and Reconstruction Error \n (mean of 2 folds for train and test datas with their own NMF) ")
fig.tight_layout()


#Now the test datas NMF is performed by using the components fitted on the train data
error_rate_train_train, error_rate_test_train, lr_result_train, compos = Proj_Type(type_proj="Train",nFold=2,n_compMax=20)

plt.style.use('default')
fig, ax1 = plt.subplots()

ax1.set_xlabel('Components')
ax1.set_ylabel('Accuracy of Logistic Regression',color="blue")
ax1.plot(range(1,20,1),lr_result_train,label="Logistic Regression Accuracy",color="blue")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  

ax2.set_ylabel(r"Error $\frac{||X-\tilde{X}V||}{||X||}$",color="red")  
ax2.plot(range(1,20,1),error_rate_train_train,color="red",label="Train Reconstruction")
ax2.plot(range(1,20,1),error_rate_test_train,color="red",linestyle=':',label="Test Reconstruction")
ax2.legend(loc="upper left")
ax2.tick_params(axis='y')
plt.title("Accuracy of LR (mean of 2 folds) \n and Reconstruction Error \n (mean of 2 folds for train and test datas with \n projection on the train datas NMF Components) ")
fig.tight_layout()




#%% Export : Go for 8 dimensions

nmf_final=decomposition.NMF(n_components=8,max_iter=1000,tol=0.001,init="random",solver="cd",random_state=1)
nmf_scaled_train1a=nmf_final.fit_transform(scaled_train_1a.iloc[:,1:])
nmf_scaled_train1a=pd.DataFrame(nmf_scaled_train1a)
nmf_scaled_train1a["label"]=scaled_train_1a.iloc[:,0]
nmf_scaled_train1a=nmf_scaled_train1a[['label',0, 1, 2, 3, 4, 5, 6, 7]]

nmf_scaled_test1a=nmf_final.transform(scaled_test_1a)
nmf_scaled_test1a=pd.DataFrame(nmf_scaled_test1a)

nmf_scaled_train1a.to_csv(data_path+r"nmf_scaled_train_1a.csv",index=False)
nmf_scaled_test1a.to_csv(data_path+r"nmf_scaled_test_1a.csv",index=False)
