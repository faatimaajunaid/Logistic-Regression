#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import numpy as np
 
def sigmoid(z):
    return 1/(1+np.exp(-z))

def mapFeature(x1,x2,degree):
    out = np.ones(len(x1))
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j)
            out= np.column_stack((out,terms))
    return out


def mapFeatureScalar(x1,x2,degree):
    out = []
    out.append(1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j)
            out.append(terms)
    return out


Lambda = 1
iterations=400
alpha=1
x1_list=[]
x2_list=[]
y_list=[]
y_outputlist=[]
y_outputlist_norm=[]

x1_rejected=[]
x2_rejected=[]
x1_accepted=[]
x2_accepted=[]

myfile = open("ex2data2.txt","r")
contents = myfile.readlines()
myfile.close()

for line in contents:
    x1,x2,y = line.split(",")
    x1_list.append(float(x1))
    x2_list.append(float(x2))
    y_list.append(float(y))
    
x1_list=np.array(x1_list)
x2_list=np.array(x2_list)
y_list=np.array(y_list)



mean1 = np.mean(x1_list)
std1 = np.std(x1_list)
x1_list_norm = (x1_list - mean1)/std1

mean2 = np.mean(x2_list)
std2 = np.std(x2_list)
x2_list_norm = (x2_list - mean2)/std2 

m=len(x1_list)

for i in range(0,m):
    if y_list[i] == 0:
        x1_rejected.append(x1_list[i])
        x2_rejected.append(x2_list[i])
    else:
        x1_accepted.append(x1_list[i])
        x2_accepted.append(x2_list[i])
    

X = mapFeature(x1_list,x2_list,6)
Y = np.transpose(np.column_stack((y_list)))
theta = np.zeros((X.shape[1],1))

for j in range(0,iterations):
    error = (-Y * np.log(sigmoid(np.dot(X,theta)))) - ((1-Y)*np.log(1-sigmoid(np.dot(X,theta))))
    cost = (1/m)*sum(error)
    cost = cost + (1/2*m)*Lambda*sum(theta**2)
    
    theta[0] = theta[0] - (alpha)*(1/m)*sum(np.dot(np.transpose(X[:,0]),(sigmoid(np.dot(X,theta))-Y)))
    
    for j in range(1,len(theta)):
        theta[j] = theta[j]*(1-alpha*(Lambda/m)) - (alpha)*(1/m)*sum(np.dot(np.transpose(X[:,j]),(sigmoid(np.dot(X,theta))-Y)))
    
   
    
    
   


plt.scatter(x1_accepted, x2_accepted,color="blue",marker="+",label="y=1")
plt.scatter(x1_rejected, x2_rejected,color="red",marker="o",label="y=0")


u_vals = np.linspace(-1,1.5,100)
v_vals= np.linspace(-1,1.5,100)
z=np.zeros((len(u_vals),len(v_vals)))
for i in range(len(u_vals)):
    for j in range(len(v_vals)):
        z[i,j] = mapFeatureScalar(u_vals[i],v_vals[j],6) @ theta

plt.contour(u_vals,v_vals,z.T,0)
        

plt.xlim([-1, 1.5])
plt.ylim([-0.8, 1.2])
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend(loc=1)


