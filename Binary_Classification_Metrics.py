import argparse
import sklearn
from sklearn import metrics
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
import pandas
import numpy as np
    
def accuracy_score(y_test,y_pred):
    count = 0
    for test, pred in zip(y_test, y_pred):
        if(test == pred):
            count = count + 1
    print("accuracy_score is:"+str(count/len(y_test)))    
    
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
    return(tp/(tp+fp))
    
def recall(y_test, y_pred):
    tp = true_positive(y_test, y_pred)
    fn = false_negative(y_test, y_pred)
    return(tp/(tp+fn))

def f1_score(y_test, y_pred):
    precision = precision(y_test, y_pred)
    recall = recall(y_test, y_pred)
    print(2*precision.recall/(precision+recall))

def ROC_AUC(y_test, y_pred):
    print("their is a complete implementation py file of ROC_AUC score & plotting of ROC Curve in Metrics_in_Machine_Learning.. check it out there:) ")

def log_loss(y_test, y_prob):
    loss = []
    epsilon = 1e-15
    for test, prob in zip(y_test, y_prob):
        prob =  np.clip(prob, epsilon, 1 - epsilon)
        p = -1*(test * np.log(prob) + (1 - test) * np.log(1 - prob))
        loss.append(p)
    print(np.mean(loss))

    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--metric",type = str)
    parser.add_argument("--model",type = str)

    dataset = pandas.read_csv("Metrics_in_Machine_Learning\\loan_modified.txt",sep = ',')

    test = dataset[dataset['source']!="train"]
    train = dataset[dataset['source']=="train"]

    X_train = train.drop(["loan_status","source","Loan_ID"],axis = 1)
    y_train = train.loan_status

    X_test = test.drop(["loan_status","source","Loan_ID"],axis = 1)
    y_test = test.loan_status

    args = parser.parse_args()

    if(args.model == "random_forest"):
            model = sklearn.ensemble.RandomForestClassifier()
    elif(args.model == "logistic_regression"):
            model = sklearn.linear_model.LogisticRegression()
    elif(args.model == "Decision_tree"):
            model = sklearn.tree.DecisionTreeClassifier()
    
    model.fit(X_train.values,y_train.values)

    y_pred = model.predict(X_test.values)
    y_prob = model.predict_proba(X_test.values)[:,1]

    if(args.metric == "accuracy_score"):
        accuracy_score(y_test, y_pred)
    elif(args.metric == "precision"):
        k = precision(y_test, y_pred)
        print(k)
    elif(args.metric == "recall"):
        k = recall(y_test, y_pred)
        print(k)
    elif(args.metric == "f1_score"):
        f1_score(y_test, y_pred)
    elif(args.metric == "ROC_AUC"):
        ROC_AUC(y_test, y_pred)
    elif(args.metric == "log_loss"):
        log_loss(y_test, y_prob)