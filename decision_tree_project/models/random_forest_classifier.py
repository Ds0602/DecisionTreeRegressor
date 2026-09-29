
from decision_tree_classifier import DecisionTreeClassifier
import random
import pandas as pd



class RandomForestClassifier:
    def __init__(self,size:int=2,max_depth:int = None, criterion:str = "gini",splitter:str = "best",max_features = None,min_samples_split:int = 2,min_samples_leaf:int = 1,ccp_alpha:int = 0,random_state:int = 0):
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

        rng = random.Random(random_state)

        for tree in self.trees:
            seed = rng.randint(0,2**32-1)
            bootstrap = features.sample(replace=True,n=len(features),random_state=seed)
            bootstrap_target = target.loc[bootstrap.index]
            #bootstrap sampling

            tree.fit(bootstrap,bootstrap_target,max_depth=max_depth,criterion=criterion,splitter=splitter,max_features=max_features,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,ccp_alpha=ccp_alpha,random_state=random_state)
        #each tree in self.trees will be fitted to its bootstrap sample

    def predict(self,features):
        predictions_data = [tree.predict(features) for tree in self.trees]
        predictions = [pd.Series([prediction_list.loc[index][0] for prediction_list in predictions_data]).mode()[0] for index in features.index]
        return pd.Series(predictions,index=features.index)
        #the final prediction will be the mode of the prediction by each tree