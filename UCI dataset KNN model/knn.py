import pandas as pd
import numpy as np
from collections import Counter
import math
'''
data1= pd.read_excel("E:\\level 4\\machine\\trainingSet.xlsx",header=None)
data2= pd.read_excel("E:\\level 4\\machine\\TestSet.xlsx",header=None)
#print(data)
x_train=data1.iloc[:,[0,1,2,3,4,5,6,7]]
print(type(x_train))
print(x_train)
#print(x_train)
y_train=data1.iloc[:,[8]]
#print(y_train[0])
x_test=data2.iloc[:,[0,1,2,3,4,5,6,7]]
#print(x_test)
y_test=data2.iloc[:,[8]]
#print(y_ttest)
'''
#convert txt file separeted by spaces to txt file seprated by comma
'''
df = pd.read_csv('E:\\level 4\\machine\\_train.txt',sep='\s+',header=None)
#print(df)
df.to_csv('E:\\level 4\\machine\\_train.txt',header=None)
print(df)

df1 = pd.read_csv('E:\\level 4\\machine\\_test.txt',sep='\s+',header=None)
#print(df)
df1.to_csv('E:\\level 4\\machine\\test.txt',header=None)
print(df)
'''
def loadData(fileName):
    df = pd.read_csv(fileName,sep=',',header=None)
    X = df[df.columns[1:9]]
    y = df[9]
    return X.values , y.values


x_train, y_train = loadData('E:\\level 4\\machine\\_train.txt')
#print(x_train)
x_test, y_test = loadData('E:\\level 4\\machine\\_test.txt')
#print(x_train[0])
def eculidenDistance(x , xi):
    d = 0.0
    for i in range(len(x)):
        d += pow(abs(x[i]-xi[i]),2)
    return math.sqrt(d)

def predict(xx,kk):
    #predict y for each column
    labels = [predictedForEachCol(i,kk) for i in xx ]
    return np.array(labels)

def predictedForEachCol(x,k):
    #compute distance between trainx andc testx
    ditance=[eculidenDistance(x,xtrain) for xtrain in x_train]
    #get k nearst sample that will sort distances ,k number of nearest neigboor for this point
    k_indices=np.argsort(ditance)[:k]
    #get the class of nearst point (y train)
    nearstLabel=[y_train[j] for j in k_indices]
    #get most common class 1 refers to first class which most common
    common=Counter(nearstLabel).most_common(1)
    return common[0][0]
'''
prdection=predict(x_test,4)
print(prdection)
accuracy=sum(prdection==y_test)/len(y_test)*100
print(accuracy)
'''
for k in range(1,10):
    print("k value = ",k)
    print("actual class = ",y_test)
    prdection = predict(x_test, k)
    print(" predicted class = " ,prdection)
    accuracy = sum(prdection == y_test) / len(y_test) * 100
    print("Number of correctly classified instances",sum(prdection == y_test) ,"Total number of instances :",len(y_test))
    print("accuracy =",accuracy)
