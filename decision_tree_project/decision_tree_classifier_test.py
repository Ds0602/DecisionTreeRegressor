from models.decision_tree_classifier import DecisionTreeClassifier
import time
import pandas as pd
from sklearn.model_selection import KFold

if False:#__name__ == "__main__":
    data = pd.read_csv("decision_tree_project/data/gender_classification_v7.csv")
    

    X = data.drop(columns=["gender"])
    y = data["gender"]




    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    predictions = []
    for train_index, test_index in kf.split(X):
    
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
    
        model = DecisionTreeClassifier(max_depth=10,criterion="gini",splitter="best")
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))
    predictions = pd.concat(predictions).sort_index()
    #cross validation

    accuracy_helper = 0
    for y_value,pred in zip(y.values,predictions.values):       
        if y_value == pred:
            accuracy_helper += 1
    accuracy = accuracy_helper / len(y)
    print(f"Accuracy: {accuracy:.4f}")

if False:#__name__ == "__main__":
    data = pd.read_csv("decision_tree_project/data/Obesity_Classification.csv")

    X = data.drop(columns=["Label","ID"])
    y = data["Label"]

    X_encoded = pd.get_dummies(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    predictions = []
    start = time.time()
    for train_index, test_index in kf.split(X_encoded):
    
        X_train, X_test = X_encoded.loc[train_index], X_encoded.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
    
        model = DecisionTreeClassifier(max_depth=10,criterion="log_loss",splitter="best")
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))
    end = time.time()
    predictions = pd.concat(predictions).sort_index()
    #cross validation

    accuracy_helper = 0
    for y_value,pred in zip(y.values,predictions.values):       
        if y_value == pred:
            accuracy_helper += 1
    accuracy = accuracy_helper / len(y)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time spent: {end-start}")

if __name__ == "__main__":
    data = pd.read_csv("decision_tree_project/data/credit_risk_dataset.csv")

    X = data.drop(columns=["loan_status"])
    y = data["loan_status"]

    X_encoded = pd.get_dummies(X,columns = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"])
    #one hot encoding cathegorical features

    kf = KFold(n_splits = 5, shuffle = True, random_state=1)
    predictions = []
    start = time.time()
    for train_index, test_index in kf.split(X_encoded):
    
        X_train, X_test = X_encoded.loc[train_index], X_encoded.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
    
        model = DecisionTreeClassifier(max_depth=10,criterion="log_loss",splitter="best",min_samples_split=10,min_samples_leaf=2,ccp_alpha=0.001)
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))
    end = time.time()
    predictions = pd.concat(predictions).sort_index()
    #cross validation

    accuracy_helper = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for y_value,pred in zip(y.values,predictions.values):       
        if y_value == pred:
            accuracy_helper += 1
        if y_value == 0:
            if pred == 0:
                true_negative += 1
            elif pred == 1:
                false_positive += 1
        elif y_value == 1:
            if pred == 1:
                true_positive += 1
            elif pred == 0:
                false_negative += 1
    accuracy = accuracy_helper / len(y)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score:  {f1_score:.4f}")
    print(f"Time spent: {end-start}")   