import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import time




class DecisionTreeClassifier:
    def __init__(self,max_depth:int = None, criterion:str = "gini",splitter:str = "best",max_features = None,min_samples_split:int = 2,min_samples_leaf:int = 1,ccp_alpha:int = 0,random_state:int = 0):
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.highest_prob_class = None
        self.highest_prob = 0
        self.skip_split = False
        self.current_error = None
        self.leaves = 0
        self.samples = 0
        self.null_direction = None
        #defining the attributes of the DecisionTree class, including left and right child nodes, split feature and value, node value

        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        random.seed(random_state)
        #if parameters are given when initializing the class, they will be used to set the attributes of the DecisionTree Class,
        #unless they are given when calling the fit method
    def __is_iterable(self,variable):
        try:
            iter(variable)
            if isinstance(variable,pd.DataFrame) and len(variable) == 1:
                return False
            return not isinstance(variable,(str,bytes,np.int64))
        except TypeError:
            return False
        #helper method used to test if a variable is iterable

    def __proba(self,target:pd.DataFrame,target_class):
        if self.__is_iterable(target):
            return (target.squeeze() == target_class).sum() / len(target)
        else:
            return 1
        #helper method which calculates the probability of a given class in the target DataFrame
    
    def __gini(self,target):        
        if self.__is_iterable(target):
            classes = target.squeeze().unique()
        else:
            classes = target
   
        gini = 1 - sum([self.__proba(target,cls)**2 for cls in classes])
        return gini
        #calculating the Gini impurity of the target DataFrame

    def __log_loss(self,target):
        if self.__is_iterable(target):
            classes = target.squeeze().unique()
            log_loss = -sum([self.__proba(target,cls) * np.log(self.__proba(target,cls)) for cls in classes])
        else:
            log_loss = -(self.__proba(target,target)) * np.log(self.__proba(target,target))
        return log_loss
        #calculating the log loss of the target DataFrame

    def __ccp_helper(self):
        if self.left_child is not None and self.right_child is not None:
            left_error, left_leaves = self.left_child.__ccp_helper()
            right_error, right_leaves = self.right_child.__ccp_helper()
            error = left_error + right_error
            leaf_number = left_leaves + right_leaves
            return (error,leaf_number)
        else:

            return (self.current_error*self.samples,1)
        #helper function used to obtain data of subtrees for ccp
        

    def __ccp(self,ccp_alpha:float):
        #cost complexity pruning
        if self.left_child is not None and self.right_child is not None:

            sub_tree_error,leaf_num = self.__ccp_helper()

            
            self.left_child.__ccp(ccp_alpha)
            self.right_child.__ccp(ccp_alpha)
            #prune children first

            
            current_alpha = (self.current_error*self.samples - sub_tree_error) / (leaf_num - 1)
            if current_alpha < ccp_alpha:
                self.left_child = None
                self.right_child = None
                self.split_feature = None
                self.split_value = None
            #pruning if current_alpha is less than or equal to ccp_alpha, which means that the cost complexity of the subtree is less than or equal to the cost complexity of the current node
    
    def fit(self,features:pd.DataFrame,target:pd.DataFrame,max_depth:int=None,criterion=None,splitter:str=None,max_features=None,min_samples_split:int=None,min_samples_leaf:int=None,ccp_alpha:int=None,random_state:int = 0):
        max_depth = self.max_depth if max_depth is None else max_depth
        criterion = self.criterion if criterion is None else criterion
        splitter = self.splitter if splitter is None else splitter
        max_features = self.max_features if max_features is None else max_features
        min_samples_split = self.min_samples_split if min_samples_split is None else min_samples_split
        min_samples_leaf = self.min_samples_leaf if min_samples_leaf is None else min_samples_leaf
        ccp_alpha = self.ccp_alpha if ccp_alpha is None else ccp_alpha
        #use the parameters if they were given when calling the fit method, else use the parameters which were given when creating the tree


            
        if self.__is_iterable(target):
            self.highest_prob_class = target.squeeze().mode()[0]
        else:
            self.highest_prob_class = target.squeeze()
        self.highest_prob = self.__proba(target,self.highest_prob_class)  





        if criterion == "gini":
            criterion_function = self.__gini                  
        elif criterion == "log_loss":
            criterion_function = self.__log_loss
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'log_loss'.")
        #selecting the criterion function based on the given parameter


        random.seed(random_state)
        minimum_split_feature = None
        minimum_split_value = None
        left_data = None
        right_data = None
        left_target = None
        right_target = None
        minimum_error = float('inf')
        self.current_error = criterion_function(target.squeeze())
        self.samples = len(features)
        #setting up variables to record the best split's data



        if self.samples < min_samples_split or max_depth == 1:
            return
        #if restrictions are met, stop splitting

        if max_features == "sqrt":
            max_features = int(round(np.sqrt(len(features.columns))))
        elif max_features == "log2":
            max_features = int(round(np.log2(len(features.columns))))
        #if max_features parameter is set to "sqrt" or "log2", calculate max_features based on total length of the dataset
        feature_columns = list(features.columns)
        if max_features is not None:
            random.shuffle(feature_columns)
            feature_columns = feature_columns[:max_features]
        #if it is given a max_features parameter, it will randomly select that number of features to consider
        


        for feature in feature_columns:

            feature_data = sorted(features[feature].dropna().unique())
            #this list will be used to calculate the values of each split point
            if len(feature_data) == 1:
                continue 
            #if there is only one value for the given feature, there would be no possible split, hence skiping this iteration
            
            if splitter == "best":
                feature_data.sort()
                split_values = [(value1 + value2)/2 for value1,value2 in zip(feature_data[1:],feature_data[:-1])]
            elif splitter == "random":
                if set(feature_data).issubset({0, 1}):
                    split_values = [0.5]
                #if the feature is binary, the only split point will be 0.5
                else:
                    min_val = np.nanmin(feature_data)
                    max_val = np.nanmax(feature_data)
                    split_values = [random.uniform(min_val, max_val) for _ in range(20)]
            #selecting split points based on the given splitter parameter
            #best: compare every possible split point and choose the best one
            #random: randomly select 20 split points in the range


            for split_value in split_values:
                #comparing each split point's error and recording the one that minimizes it

                left_data = features[features[feature] <= split_value]
                right_data = features[features[feature] > split_value]

                nan_data = features[pd.isna(features[feature])]
                #splitting data into left_data, right_data and nan_data if there are nan values
                for null_direction in ["left","right"]:
                    if null_direction == "left":
                        left_data = pd.concat([left_data,nan_data])
                    elif null_direction == "right":
                        right_data = pd.concat([right_data,nan_data])
                    #try out which direction would be better for samples with nan values

                    left_target = pd.DataFrame(target.loc[left_data.index], index=left_data.index)
                    right_target = pd.DataFrame(target.loc[right_data.index], index=right_data.index)
                    #dividing target into left and right target

                    if len(left_target) < min_samples_leaf or len(right_target) < min_samples_leaf:
                        continue
                    #if the length of samples in one of the leafs is less than min_samples_leaf parameter, then it won't be considered


                    error = criterion_function(left_target.squeeze()) * len(left_target) + criterion_function(right_target.squeeze()) * len(right_target)
                    #calculating the error of the split point based on the criterion function

                    if error < minimum_error and error < self.current_error*self.samples:
                        minimum_left_data = left_data
                        minimum_right_data = right_data
                        minimum_left_target = left_target
                        minimum_right_target = right_target

                    
                        minimum_error = error
                        minimum_split_feature = feature
                        minimum_split_value = split_value


                        self.null_direction = null_direction
                        #recording the data of the split point that minimizes error


        if minimum_split_feature is None:
            return 
        #if no split was found, return
        elif max_depth is None or max_depth > 1:
            self.split_feature = minimum_split_feature
            self.split_value = minimum_split_value
            
            
            self.left_child = DecisionTreeClassifier()
            self.right_child = DecisionTreeClassifier()

            if pd.isna(max_depth):
                self.left_child.fit(minimum_left_data, minimum_left_target,criterion=criterion,
                                    splitter=splitter,max_features=max_features,min_samples_split=min_samples_split)
                self.right_child.fit(minimum_right_data, minimum_right_target,criterion=criterion,
                                    splitter=splitter,max_features=max_features,min_samples_split=min_samples_split)
            elif max_depth > 1:
                self.left_child.fit(minimum_left_data, minimum_left_target, max_depth=(max_depth - 1),
                                    criterion=criterion,splitter=splitter,max_features=max_features,min_samples_split=min_samples_split)
                self.right_child.fit(minimum_right_data, minimum_right_target, max_depth=(max_depth - 1),
                                    criterion=criterion,splitter=splitter,max_features=max_features,min_samples_split=min_samples_split)                       
        #if can split, create left and right child nodes and fit them recursively with the remaining depth

        if ccp_alpha < 0:
            raise ValueError("ccp_alpha must be greater than 0")
        elif ccp_alpha > 0:
            self.__ccp(ccp_alpha)
        #cost complexity pruning

    def __row_predict(self,row:pd.DataFrame):

        if self.split_feature is None:
            return self.highest_prob_class           
        elif row[self.split_feature] <= self.split_value or (pd.isna(row[self.split_feature]) and self.null_direction == "left"):
            return self.left_child.__row_predict(row)
        elif row[self.split_feature] > self.split_value or (pd.isna(row[self.split_feature]) and self.null_direction == "right"):
            return self.right_child.__row_predict(row)
        #predict the value for a single row based on the split feature and value, recursively calling the child nodes

        
    def predict(self,features:pd.DataFrame):
        predictions = {i: self.__row_predict(features.loc[i]) for i in features.index}
        return pd.DataFrame(predictions.values(), index=predictions.keys())    
        #predict the values for all rows in the given features DataFrame by calling __row_predict for each row
    

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
    
        model = DecisionTreeClassifier(max_depth=10,criterion="log_loss",splitter="best",min_samples_split=10,min_samples_leaf=2,ccp_alpha=0.0001)
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
