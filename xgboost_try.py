import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb

def loadDataFrame():
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    iris_df = pd.DataFrame(iris_data, columns=iris.feature_names)
    iris_df['target'] = pd.Series(iris_target)

    return iris_df


if __name__ == "__main__":
    iris_df = loadDataFrame()
    value_counts = iris_df['target'].value_counts()
    print(value_counts)
    X_iris = iris_df.iloc[:, :-1].values
    y_iris = iris_df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, stratify=y_iris)

    model = LGBMClassifier()

    model.fit(X_train, y_train)
    print(model.get_params())

    y_pred = model.predict(X_test)
    print("accuracy: %s" % np.mean(y_pred == y_test))

