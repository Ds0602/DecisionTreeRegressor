# Decision Tree (From Scratch)

This project implements decision trees built from scratch using Python. It does not rely on libraries like `scikit-learn` for the model itself, and is mainly for learning and experimentation purposes. 

## Models
* Decision tree regressor
* Decision tree classifier
* Random forest regressor
* Random forest classifier

## Features

* Supports regression using:
  * Regression:
    * Mean Squared Error (`squared_error`)
    * Mean Absolute Error (`absolute_error`)
  * Classification:
    * Gini impurity (`gini`)
    * Log loss (`log_loss`)
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

Decision tree models build a binary tree by recursively splitting the dataset based on the feature and split value that minimizes error.

Each node:

* Stores a prediction value (Regression: mean or median of target/Classification: most probable class)
* Chooses the best split based on the selected criterion
* Stops splitting based on conditions like depth or sample size

Pruning is done after training using cost-complexity pruning.

## Usage example
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
* `criterion`: Regression: `"squared_error"` or `"absolute_error"` Classification: `"gini"` or `"log_loss"`
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

The datasets used in this project come from Kaggle.

* Source: Kaggle laptop dataset
 * File used: `decision_tree_project/data/laptop_data.csv`
 * link: https://www.kaggle.com/datasets/muhammadmusharraf444/laptop-specifications-and-price-prediction-dataset/data
 
* Source: Kaggle Obesity classification dataset
 * File used: `decision_tree_project/data/Obesity_Classification.csv`
 * link: https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset
 
* Source: Kaggle gender classification dataset
 * File used: `decision_tree_project/data/gender_classification.csv`
 * link: https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset
 
* Source: Kaggle credit risk dataset
 * File used: `decision_tree_project/data/credit_risk_dataset.csv`
 * link: https://www.kaggle.com/datasets/laotse/credit-risk-dataset


## Folders

* `models`: main implementation of decision trees
* `data`: folder where datasets are

## License

This project is for educational use. Feel free to modify and experiment.
