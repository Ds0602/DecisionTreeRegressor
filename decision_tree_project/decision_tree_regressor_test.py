from models.decision_tree_regressor import DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time

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


data = pd.read_csv("decision_tree_project/data/laptop_data.csv")

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



    
#cross validation to evaluate the model's performance using RMSE and MAE metrics, and also measuring the time taken for training and prediction

kf = KFold(n_splits=5, shuffle=True, random_state=1)
    #splitting data into 5 folds for cross-validation, shuffling the data before splitting to ensure randomness

start = time.time()

    
predictions = []
for train_index, test_index in kf.split(X_encoded):

    X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = DecisionTreeRegressor(max_depth=15,criterion="squared_error",splitter="best",max_features=250,min_samples_split=10,min_samples_leaf=2,random_state = 1)
    model.fit(X_train, y_train)
    predictions.append(model.predict(X_test))
#cross validation

end = time.time()
time_taken = end - start

predictions = pd.concat(predictions).sort_index()
current_rmse = mse(y.values, predictions.values)**0.5
current_mae = mae(y.values, predictions.values)
target_mean = np.mean(y.values)
prediction_mean = np.mean(predictions.values)
percentage_mae = current_mae / target_mean * 100
percentage_rmse = current_rmse / target_mean * 100

print(f"RMSE: {current_rmse}")
print(f"MAE: {current_mae}")
print(f"Percentage MAE: {percentage_mae}%")
print(f"Percentage RMSE: {percentage_rmse}%")
print(f"Target mean: {target_mean}")
print(f"Prediction mean: {prediction_mean}")
print(f"Percentage difference: {abs(target_mean - prediction_mean)/target_mean*100:.2f}%")
print(f"Time taken: {time_taken} seconds")
print("-----------------------------")
errors = {10:0,20:0,30:0,40:0,50:0,60:0,70:0,80:0,90:0,100:0,">100":0}
for prediction,target in zip(predictions.values,y.values):
    error = float((abs(prediction-target)/target*100)[0])
    errors[10*(int(error/10)+1) if error <100 else ">100"] += 1
for n in errors.keys():
    if n == ">100":
        print(f"Number of predictions with error greater than 100%: {errors[n]}")
        print(f"Percentage of predictions with error greater than 100%: {errors[n]/len(y)*100:.2f}%")
        print()
        continue
    print(f"Number of predictions with error between {n-10}% and {n}%: {errors[n]}")
    print(f"Percentage of predictions with error between {n-10}% and {n}%: {errors[n]/len(y)*100:.2f}%")
    print()
