import pandas
import argparse
import numpy
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#there are 3 Types of precision in case of Multi-class classification. 
#1. Macro averaged precision
#2. Micro averaged precision
#3. Weighted precision

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

def precision(y_test, y_pred):
    tp =true_positive(y_test, y_pred)
    fp = false_positive(y_test, y_pred)
    try:
        return(tp/(tp+fp))
    except ZeroDivisionError:
        return 0

def Macro_averaged_precision(y_test, predictions):
    precisions = []
    for i in range(1,5):
        temp_ytest = [1 if x == i else 0 for x in y_test]
        temp_ypred = [1 if x == i else 0 for x in predictions]
        print(temp_ypred)
        print(temp_ytest)
        prec = precision(temp_ytest, temp_ypred)
        precisions.append(prec)
    
    return (sum(precisions)/len(precisions))
         
def Micro_averaged_precision(y_test, predictions):
    tp = 0
    fp = 0
    for i in range(1,5):
        temp_ytest = [1 if x == i else 0 for x in y_test]
        temp_ypred = [1 if x == i else 0 for x in predictions]

        tp += true_positive(temp_ytest, temp_ypred)
        fp += false_positive(temp_ytest, temp_ypred)

    precisions = tp / (tp + fp)

    return precisions

def weighted_precision(y_test, predictions):
    num_classes = len(numpy.unique(y_test))
    #coutns for every class
    precision = 0
    for i in range(1, num_classes):
        temp_ytest = [1 if x == i else 0 for x in y_test]
        temp_ypred = [1 if x == i else 0 for x in predictions]

        tp = true_positive(temp_ytest, temp_ypred)
        fp = false_positive(temp_ytest, temp_ypred)
        
        try:
            preai = tp / (tp+fp)
        except ZeroDivisionError:
            preai = 0

        weighted = preai*sum(temp_ytest)

        precision += weighted

    precision = precision/len(y_test)
    return precision

if __name__ == "__main__":
    
    data = pandas.read_csv("C:\\Users\\iamvi\\OneDrive\\Desktop\\Metrics_in_Machine_Learning\\development-index\\Development Index.csv")
    
    train = data.drop(['Development Index'], axis = 1).values
    test = data["Development Index"].values

    model = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(train, test, stratify = test)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Macro precision is:", Macro_averaged_precision(y_test, predictions))
    print("Micro precision is:", Micro_averaged_precision(y_test, predictions))
    print("Weighted precision is:", weighted_precision(y_test, predictions))
    print("sklearn Macro", metrics.precision_score(y_test, predictions, average = "macro"))
    print("sklearn Micro", metrics.precision_score(y_test, predictions, average = "micro"))
    print("sklearn weighted", metrics.precision_score(y_test, predictions, average = "weighted"))