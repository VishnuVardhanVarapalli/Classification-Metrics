import pandas
import argparse
import numpy
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#there are 3 Types of recall in case of Multi-class classification. 
#1. Macro averaged recall
#2. Micro averaged recall
#3. Weighted recall

def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp
    
def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn
    
def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp
    
def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

def recall(y_test, y_pred):
    tp = true_positive(y_test, y_pred)
    fn = false_negative(y_test, y_pred)
    return(tp/(tp+fn))

def Macro_averaged_recall(y_test, predictions):
    recalls = []
    for i in range(1,5):
        temp_ytest = [1 if x == i else 0 for x in y_test]
        temp_ypred = [1 if x == i else 0 for x in predictions]
        print(temp_ypred)
        print(temp_ytest)
        rec = recall(temp_ytest, temp_ypred)
        recalls.append(rec)
    
    return (sum(recalls)/len(recalls))
         
def Micro_averaged_recall(y_test, predictions):
    tp = 0
    tn = 0
    for i in range(1,5):
        temp_ytest = [1 if x == i else 0 for x in y_test]
        temp_ypred = [1 if x == i else 0 for x in predictions]

        tp += true_positive(temp_ytest, temp_ypred)
        tn += true_negative(temp_ytest, temp_ypred)

    recall = tp / (tp + tn)

    return recall

def weighted_recall(y_test, predictions):
    num_classes = len(numpy.unique(y_test))
    #counts for every class
    recall = 0
    for i in range(1, num_classes):
        temp_ytest = [1 if x == i else 0 for x in y_test]
        temp_ypred = [1 if x == i else 0 for x in predictions]

        tp = true_positive(temp_ytest, temp_ypred)
        tn = true_negative(temp_ytest, temp_ypred)
        
        try:
            rec = tp / (tp+tn)
        except ZeroDivisionError:
            rec = 0

        weighted = rec*sum(temp_ytest)

        recall += weighted

    recall = recall/len(y_test)
    return recall

if __name__ == "__main__":
    
    data = pandas.read_csv("C:\\Users\\iamvi\\OneDrive\\Desktop\\Metrics_in_Machine_Learning\\development-index\\Development Index.csv")
    
    train = data.drop(['Development Index'], axis = 1).values
    test = data["Development Index"].values

    model = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(train, test, stratify = test)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Macro recall is:", Macro_averaged_recall(y_test, predictions))
    print("Micro recall is:", Micro_averaged_recall(y_test, predictions))
    print("Weighted recall is:", weighted_recall(y_test, predictions))
    print("sklearn Macro", metrics.recall_score(y_test, predictions, average = "macro"))
    print("sklearn Micro", metrics.recall_score(y_test, predictions, average = "micro"))
    print("sklearn weighted", metrics.recall_score(y_test, predictions, average = "weighted"))