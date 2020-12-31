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

scaled_test_1a=pd.read_csv(r"C:\Users\rapha\Desktop\GrandeDimProjet\EEG\scaled_test_1a.csv")
scaled_train_1a=pd.read_csv(r"C:\Users\rapha\Desktop\GrandeDimProjet\EEG\scaled_train_1a.csv")

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


#Use the CV grid to gest the best couple of parameters to predict, with 6 components


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

#Let's just try our "best performance" for lr prediction while allowing a high number of iterations

#%% Choosing the number of components for a prediction : nmf_6

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
    
    kf = KFold(n_splits=5,shuffle=True)
    
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

fig.tight_layout()


#%% Export : Go for 8 dimensions

nmf_final=decomposition.NMF(n_components=8,max_iter=1000,tol=0.001,init="random",solver="cd",random_state=1)
nmf_scaled_train1a=nmf_final.fit_transform(scaled_train_1a.iloc[:,1:])
nmf_scaled_train1a=pd.DataFrame(nmf_scaled_train1a)
nmf_scaled_train1a["label"]=scaled_train_1a.iloc[:,0]
