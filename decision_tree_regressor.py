import random
import pandas as pd
import time
import numpy as np




def convert_memory(data):
            if "TB" in data:
                return  float(data[:-2]) * 1000
            else:
                return int(data[:-2])

def mse(target, prediction):
    #mean squared error
    if isinstance(prediction,np.float64) or isinstance(target,np.float64):
            return (target-prediction)**2
    return sum([(x-y)**2 for x,y in zip(prediction,target)]) / len(prediction)

def mae(target, prediction):
    #mean absolute error
    if isinstance(prediction,np.float64) or isinstance(target,np.float64):
        return abs(target-prediction)
    return sum([abs(x-y) for x,y in zip(prediction,target)])/len(prediction)



class DecisionTreeRegressor:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.value = None
        self.single_value_feature = False
        self.skip_split = False
        self.current_error = None
        self.leaves = 0
        self.samples = 0
        #defining the attributes of the DecisionTree class, including left and right child nodes, split feature and value, node value, and a flag for single-value features

    def __mse(self,data):
        #mean squared error
        try:
            mean = sum(data)/len(data)
            error_sum = sum([(x-mean)**2 for x in data])

            return error_sum/len(data)
        except:
            return 0
    def __mae(self,data):
        #mean absolute error
        try:
            median = np.median(data)
            error_sum = sum([abs(x-median) for x in data])

            return error_sum/len(data)
        except:
            return 0

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

            self.left_child.__ccp(ccp_alpha)
            self.right_child.__ccp(ccp_alpha)
            #prune children first

            sub_tree_error,leaf_num = self.__ccp_helper()
            if leaf_num <= 1:
                return #avoiding bugs
            
            current_alpha = (self.current_error*self.samples - sub_tree_error) / (leaf_num - 1)
            if current_alpha <= ccp_alpha:
                self.left_child = None
                self.right_child = None
                self.split_feature = None
                self.split_value = None
            #pruning
    
    def fit(self,features:pd.DataFrame,target:pd.DataFrame,max_depth:int=None,criterion="squared_error",splitter="best",max_features=None,min_samples_split:int=2,min_samples_leaf:int=1,ccp_alpha:int=0,random_state:int = 0):

        if criterion == "squared_error":
            self.value = target.squeeze().mean()                    
        elif criterion == "absolute_error":
            self.value = np.median(target.squeeze())

        if criterion == "squared_error":
            criterion_function = self.__mse
        elif criterion == "absolute_error":
            criterion_function = self.__mae
        else:
            raise ValueError("Invalid criterion. Use 'squared_error' or 'absolute_error'.")
        #selecting the criterion function based on the given parameter

        random.seed(random_state)
        minimum_split_feature = None
        minimum_split_value = None
        left_data = None
        right_data = None
        left_target = None
        right_target = None
        minimum_error = float('inf')
        self.current_error = criterion_function(target.to_numpy().squeeze())
        self.samples = len(features)
        #setting up variables to record the best split's data

        if self.samples < min_samples_split or max_depth == 1 or target.nunique() == 1: # or minimum_split_feature is None:# or len(minimum_left_data) == 0 or len(minimum_right_data) == 0:
            return

        feature_columns = list(features.columns)
        if max_features is not None:
            random.shuffle(feature_columns)
            feature_columns = feature_columns[:max_features]
        #if it is given a max_features parameter, it will randomly select that number of features to consider
        



        for feature in feature_columns:

            feature_data = list(set(features[feature]))
            #this list will be used to calculate the values of each split point

            if len(feature_data) == 1:
                continue

            if splitter == "best":
                feature_data.sort()
                split_values = [(value1 + value2)/2 for value1,value2 in zip(feature_data[1:],feature_data[:-1])]
            elif splitter == "random":
                if set(feature_data).issubset({0, 1}):
                    split_values = [0.5]
                else:
                    min_val = np.nanmin(feature_data)
                    max_val = np.nanmax(feature_data)
                    split_values = [random.uniform(min_val, max_val) for _ in range(10)]
            #selecting split points based on the given splitter parameter
            #best: compare every possible split point and choose the best one
            #random: randomly select 10 split points in the range


            for split_value in split_values:
                #comparing each split point's error and recording the one that minimizes it

                left_data = features[features[feature] <= split_value]
                right_data = features[features[feature] > split_value]

                nan_data = features[pd.isna(features[feature])]
                left_data = pd.concat([left_data,nan_data])

                left_target = target.loc[left_data.index]
                right_target = target.loc[right_data.index]
                #splitting the data into left and right based on the split point

                if len(left_target) < min_samples_leaf or len(right_target) < min_samples_leaf:
                    continue
                #if the length of samples in one of the leafs is less than min_samples_leaf parameter, then it won't be considered


                error = criterion_function(left_target.to_numpy().squeeze()) * len(left_target) + criterion_function(right_target.to_numpy().squeeze()) * len(right_target)
                #calculating the error of the split point based on the criterion function

                if error < minimum_error and error < self.current_error*self.samples:
                    minimum_left_data = left_data
                    minimum_right_data = right_data
                    minimum_left_target = left_target
                    minimum_right_target = right_target

                    
                    minimum_error = error
                    minimum_split_feature = feature
                    minimum_split_value = split_value
                    self.single_value_feature = False
                #recording the data of the split point that minimizes error


        if minimum_split_feature is None:
            return 
        elif max_depth is None or max_depth > 1:
            self.split_feature = minimum_split_feature
            self.split_value = minimum_split_value
            
            
            self.left_child = DecisionTreeRegressor()
            self.right_child = DecisionTreeRegressor()

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
        

    def __row_predict(self,row:pd.DataFrame):

        if self.split_feature is None:
            return self.value            
        elif row[self.split_feature] <= self.split_value or pd.isna(row[self.split_feature]):
            return self.left_child.__row_predict(row)
        elif row[self.split_feature] > self.split_value:
            return self.right_child.__row_predict(row)
        #predict the value for a single row based on the split feature and value, recursively calling the child nodes
        #if the value of the split feature is Nan or None, then randomly choose a child
        
    def predict(self,features:pd.DataFrame):
        predictions = {i: self.__row_predict(features.loc[i]) for i in features.index}
        return pd.DataFrame(predictions.values(), index=predictions.keys())    
        #predict the values for all rows in the given features DataFrame by calling __row_predict for each row


if __name__ == "__main__":
    data = pd.read_csv("data/laptop_data (1).csv")

    screen_type = []
    resolution = []
    for resolution_data in data["ScreenResolution"]:
        resolution_data = resolution_data.split(" ")
        resolution_amount = resolution_data[-1].split("x")

        if len(resolution_data) == 1:
            screen_type.append(None)
        else:
            screen_type.append(" ".join(resolution_data[:-1]))

        resolution.append(int(resolution_amount[0])*int(resolution_amount[1]))

    cpu_type = []
    cpu_frequency = []
    for cpu_data in data["Cpu"]:
        cpu_type_data = cpu_data.split(" ")
        cpu_type.append(" ".join(cpu_type_data[:-1]))
        cpu_frequency.append(float(cpu_type_data[-1][:-3]))


    ram_size = []
    for ram_data in data["Ram"]:
        ram_size.append(int(ram_data[:-2]))

    weight = []
    for weight_data in data["Weight"]:
        weight.append(float(weight_data[:-2]))


    memory_size = []
    memory_type = []
    additional_memory = []
    add_memory_size = []
    add_memory_type = []

    for memory_data in data["Memory"]:
        if "+" in memory_data:
            additional_memory.append("True")

            memory_data, add_memory_data = memory_data.split("+")

            memory_data = memory_data.strip().split(" ")
            add_memory_data = add_memory_data.strip().split(" ")

            memory_size.append(convert_memory(memory_data[0]))
            memory_type.append(" ".join(memory_data[1:]))
            add_memory_size.append(convert_memory(add_memory_data[0]))
            add_memory_type.append(" ".join(add_memory_data[1:]))

        else:
            additional_memory.append("False")
            memory_data = memory_data.split(" ")
            
            memory_size.append(convert_memory(memory_data[0]))
            memory_type.append(" ".join(memory_data[1:]))
            add_memory_size.append(None)
            add_memory_type.append(None)


    data["ScreenType"] = screen_type
    data["Resolution"] = resolution
    data["CpuType"] = cpu_type
    data["CpuFrequency"] = cpu_frequency
    data["RamSize"] = ram_size
    data["MemorySize"] = memory_size
    data["MemoryType"] = memory_type
    data["AdditionalMemory"] = additional_memory
    data["AdditionalMemorySize"] = add_memory_size
    data["AdditionalMemoryType"] = add_memory_type
    data["WeightInt"] = weight
    #In the code above, I handled data so that I could quantify some of the features such as Resolution, CpuFrequency, RamSize, etc.
    #while maintaining some categorical features, such as ScreenType, CpuType, MemoryType, etc.

    features = ["Company", "TypeName", "Inches", "ScreenType","Resolution", "RamSize", "MemorySize","MemoryType",
                "AdditionalMemory","AdditionalMemorySize","AdditionalMemoryType", "WeightInt", "CpuType","CpuFrequency", "Gpu", "OpSys"]

    X = data[features]
    y = data["Price"]
    X_encoded = pd.get_dummies(X, columns=["Company","TypeName","ScreenType","MemoryType","AdditionalMemory","AdditionalMemoryType","CpuType","Gpu","OpSys"])
    #encoding categorical features, while maintaining numerical features

    start = time.time()

    model = DecisionTreeRegressor()
    model.fit(X_encoded, y, max_depth=15,criterion="squared_error",splitter="random",max_features=200,min_samples_split=50,min_samples_leaf=10,random_state = 1)
    predictions = model.predict(X_encoded)
    #creating the model, fitting data, and making predictions

    end = time.time()
    

    for target, prediction in zip(y.head(100).values, predictions.head(100).values):
            
        print(f"Target: {target}, Prediction: {prediction[0]}, Error: {mae(target, prediction)}\n")

    print(f"time spent:{end-start}")
    print(f"RMSE: {mse(y.head(1000).values, predictions.head(1000).values)**0.5}")
    print(f"MAE: {mae(y.head(1000).values, predictions.head(1000).values)}")
