
import pandas as pd
import sys
#import graphviz
import matplotlib.pyplot as plt
import statistics as sc
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn import preprocessing
 




col_names = ['target', "issue1","issue2","issue3","issue4","issue5","issue6","issue7","issue8","issue9","issue10","issue11","issue12","issue13","issue14","issue15","issue16"]
 

def delmiss(x): 
    for i in col_names :
        tmp = x[i].tolist()
        y = tmp.count('y')
        n = tmp.count('n')
        for j in range (len(tmp)):
            if tmp[j] == '?' and y > n :
                tmp[j] = 'y'
            elif tmp[j] == '?' and y<=n :
                tmp[j] = 'n'
        x[i] = tmp     
    return x

housevotes= pd.read_excel('house-votes-84.data1.xlsx',header=None,names=col_names)

housevotes = delmiss(housevotes)

le = preprocessing.LabelEncoder()

for i in col_names:
    
         housevotes[i]=le.fit_transform(housevotes[i])


feature_cols = ["issue1","issue2","issue3","issue4","issue5","issue6","issue7","issue8","issue9","issue10","issue11","issue12","issue13","issue14","issue15","issue16"]
X = housevotes[feature_cols] # Features
y = housevotes.target # Target variable


counter=0
  #function to get Desicion tree with 25% trainning and 75% testing  
def getDS(x,y):
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75) # 25% training and 75% test
# Create Decision Tree classifer object
     clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
     clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
     y_pred = clf.predict(X_test)

 # Model Accuracy, how often is the classifier correct?
     treeObj = clf.tree_
     print ("Size of tree",treeObj.node_count)
     print("Accuracy:  %.2f%%" % (metrics.accuracy_score(y_test, y_pred)*100))
     print("------------------------------------------------------------")
 
    #run function 3 times to report different between size of tree and accuracy each iteration
while counter<3:   
      getDS(X,y)
      counter=counter+1


    

maximumAcc=[]
minimumAcc=[]
avgAcc=[]
maxTSize=[]
minTSize=[]
avgTSize=[]
trainRange=[0.3,0.4,0.5,0.6,0.7]


for i in range(len(trainRange)):
         Accuracies=[]
         treeSizes=[]
         for j in range(5):
    # Split dataset into training set and test set
             X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainRange[i]) 
    # Create Decision Tree classifer object
             clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
             clf = clf.fit(X_train,y_train)
       

    #Predict the response for test dataset
             y_pred = clf.predict(X_test)

     # Model Accuracy, how often is the classifier correct?
             treeObj = clf.tree_
             
             treeSizes.append(treeObj.node_count)
            # print ("Size of tree", treeObj.node_count)
            # print("Sizes",sys.getsizeof(X_test),sys.getsizeof(X_train))
             Accuracies.append(round(metrics.accuracy_score(y_test, y_pred)*100,2))
            # print("Accuracy%d:  %.2f%%" % (j+1,metrics.accuracy_score(y_test, y_pred)*100))
            
            
         '''dot_data = tree.export_graphviz(clf, out_file=None)
         graph = graphviz.Source(dot_data)
         graph.render("imgdata")
         fig = plt.figure(figsize=(25,20))
         f = tree.plot_tree(clf, filled=True)'''
        
      

         print("Training Size is {}{}".format(int(trainRange[i]*100),'%'))    
         maxAcc = max(Accuracies)
         print("Maximum accuracy is : " , maxAcc)
         maximumAcc.append(maxAcc)
         minAcc = min(Accuracies)
         print("minimum accuracy is : " , minAcc)
         #minimumAcc.append(minAcc)
         avgAcc = sc.mean(Accuracies)
         print("mean accuracy is : ", avgAcc)
         #averageAcc.append(avgAcc)
         mxTS = max(treeSizes)
         print("Maximum treeSizes is : " , mxTS)
         #maxTSize.append(mxTS)
         mnTS = min(treeSizes)
         print("minimum treeSizes is : " , mnTS)
         #minTSize.append(mnTS)
         avTS = sc.mean(treeSizes)
         print("mean treeSizes is : ", avTS)
         #avgTSize.append(avTS)
         print("------------------------------------------------------------")
         
         
plt.plot(Accuracies, label='Accuracies')
plt.plot(treeSizes , label='TreeSizes')
         
         
         
         

         

