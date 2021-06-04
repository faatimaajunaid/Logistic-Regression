#!/usr/bin/env python
# coding: utf-8

# In[92]:


import matplotlib.pyplot as plt
import numpy as np
 
def sigmoid(z):
    return 1/(1+np.exp(-z))

iterations=400
alpha=1
x1_list=[]
x2_list=[]
y_list=[]
y_outputlist=[]
y_outputlist_norm=[]

x1_notadmitted=[]
x2_notadmitted=[]
x1_admitted=[]
x2_admitted=[]

myfile = open("ex2data1.txt","r")
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
        x1_notadmitted.append(x1_list[i])
        x2_notadmitted.append(x2_list[i])
    else:
        x1_admitted.append(x1_list[i])
        x2_admitted.append(x2_list[i])
    

X = np.column_stack((np.ones(m),x1_list_norm,x2_list_norm))
Y = np.transpose(np.column_stack((y_list)))
theta = np.zeros((3,1))

for j in range(0,iterations):
    error = (-Y * np.log(sigmoid(np.dot(X,theta)))) - ((1-Y)*np.log(1-sigmoid(np.dot(X,theta))))
    cost = (1/m)*sum(error)
    theta = theta - alpha*(1/m)*np.dot(np.transpose(X),sigmoid(np.dot(X,theta))-Y)
    
    
    
for k in range(0,m):
    temp = -(theta[0] + theta[1]*x1_list_norm[k])/theta[2]
    y_outputlist_norm.append(temp)
   

for k in range(0,m):
    y_outputlist.append(y_outputlist_norm[k]*std2 + mean2)
   

plt.scatter(x1_admitted, x2_admitted,color="blue",marker="+",label="Admitted")
plt.scatter(x1_notadmitted, x2_notadmitted,color="red",marker="o",label="Not Admitted")

#plt.show()
plt.plot(x1_list,y_outputlist,color="green")
#plt.show()

plt.xlim([30, 100])
plt.ylim([30, 100])
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)


x_test=[]
x1_val = 45
x2_val = 85
x_test.append(1)
x_test.append((x1_val-mean1)/std1)
x_test.append((x2_val-mean2)/std2)

x_test = np.array(x_test)
probability = sigmoid(np.dot(x_test,theta))
print(probability)

