from decision_tree_classifier import DecisionTreeClassifier
import random
import pandas as pd
import time
from sklearn.model_selection import  KFold


class RandomForestClassifier:
    def __init__(self,size:int=3,max_depth:int = None, criterion:str = "gini",splitter:str = "best",max_features = None,min_samples_split:int = 2,min_samples_leaf:int = 1,ccp_alpha:int = 0,random_state:int = 0):
        self.trees = [DecisionTreeClassifier(max_depth=max_depth,criterion=criterion,splitter=splitter,max_features=max_features,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,ccp_alpha=ccp_alpha,random_state = random_state) for _ in range(size)]
        self.size = size
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        #if parameters are given when initializing the class, they will be used to set the attributes of the DecisionTree Class,
        #unless they are given when calling the fit method

    def fit(self,features,target,max_depth:int = None, criterion:str = None,splitter:str = None,max_features = None,min_samples_split:int = None,min_samples_leaf:int = None,ccp_alpha:int = None,random_state:int = None):
        max_depth = self.max_depth if max_depth is None else max_depth
        criterion = self.criterion if criterion is None else criterion
        splitter = self.splitter if splitter is None else splitter
        max_features = self.max_features if max_features is None else max_features
        min_samples_split = self.min_samples_split if min_samples_split is None else min_samples_split
        min_samples_leaf = self.min_samples_leaf if min_samples_leaf is None else min_samples_leaf
        ccp_alpha = self.ccp_alpha if ccp_alpha is None else ccp_alpha
        random_state = self.random_state if random_state is None else random_state
        #use the parameters if they were given when calling the fit method, else use the parameters which were given when creating the tree
        random.seed(random_state)

        data_size = len(features)//self.size
        index_data = list(features.index)
        random.shuffle(index_data)
        for n,tree in zip(range(self.size),self.trees):
            if n == self.size-1:
                index = index_data[n*data_size:]
            else:
                index = index_data[n*data_size:(n+1)*data_size]
            tree.fit(features.loc[index],target.loc[index],max_depth=max_depth,criterion=criterion,splitter=splitter,max_features=max_features,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,ccp_alpha=ccp_alpha,random_state=random_state)
        #dividing the data into "self.size" subsets and making "self.size" trees fit each subset

    def predict(self,features):
        predictions_data = [tree.predict(features) for tree in self.trees]

        predictions = [pd.Series([prediction_list.loc[index][0] for prediction_list in predictions_data]).mode()[0] for index in features.index]
        return pd.Series(predictions,index=features.index)
        #the final prediction will be the mode of the prediction by each tree

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
    
        model = RandomForestClassifier(size=5,max_depth=10,criterion="log_loss",splitter="best",min_samples_split=10,min_samples_leaf=2,ccp_alpha=0.001,random_state=1)
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