# Decision Tree Regressor (From Scratch)

This project is a basic implementation of a Decision Tree Regressor built from scratch using Python. It does not rely on libraries like `scikit-learn` for the model itself, and is mainly for learning and experimentation purposes.

## Features

* Supports regression using:

  * Mean Squared Error (`squared_error`)
  * Mean Absolute Error (`absolute_error`)
* Two splitting strategies:

  * `best` (checks all possible split points)
  * `random` (uses random split points)
* Handles:

  * Numerical and categorical (encoded) data
  * Missing values (NaNs are sent to the left branch)
* Includes Cost Complexity Pruning (`ccp_alpha`)
* Basic control over:

  * `max_depth`
  * `min_samples_split`
  * `min_samples_leaf`
  * `max_features`

## How It Works

The model builds a binary tree by recursively splitting the dataset based on the feature and split value that minimizes error.

Each node:

* Stores a prediction value (mean or median of target)
* Chooses the best split based on the selected criterion
* Stops splitting based on conditions like depth or sample size

Pruning is done after training using cost-complexity pruning.

## Usage

```python
from decision_tree_regressor import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(
    features=X, 
    target=y,
    max_depth=5,
    criterion="squared_error",
    splitter="best",
    min_samples_split=2,
    min_samples_leaf=1,
    ccp_alpha=0
)
```

## Parameters

* `max_depth`: Maximum depth of the tree
* `criterion`: `"squared_error"` or `"absolute_error"`
* `splitter`: `"best"` or `"random"`
* `max_features`: Number of features to consider at each split
* `min_samples_split`: Minimum samples required to split
* `min_samples_leaf`: Minimum samples required in a leaf
* `ccp_alpha`: Pruning strength
* `random_state`: Seed for reproducibility

## Notes

* This implementation is not optimized for speed.
* It is mainly intended for understanding how decision trees work internally.
* One-hot encoded categorical features may behave differently depending on the splitter.

## Requirements

* Python 3.x
* pandas
* numpy

## Dataset

The dataset used in this project comes from Kaggle.

* Source: Kaggle laptop dataset
* File used: `data/laptop_data (1).csv`
*link: https://www.kaggle.com/datasets/muhammadmusharraf444/laptop-specifications-and-price-prediction-dataset/data


## File

* `decision_tree_regressor.py`: main implementation
* `laptop_data (1).csv`: data set used for testing

## License

This project is for educational use. Feel free to modify and experiment.
