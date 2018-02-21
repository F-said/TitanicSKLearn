from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def find_best_thresh(X, y, X_c, y_c, classifier):
    thresh_list = np.arange(0, 1, 0.05)
    accuracy_list = []
    best_accuracy = 0
    best_t = thresh_list[0]

    for th in thresh_list:
        thresh_selection = SelectFromModel(estimator=classifier, threshold=th)
        thresh_selection.fit(X, y)
        X = thresh_selection.transform(X)

        classifier.fit(X, y)

        X_c = thresh_selection.transform(X_c)
        y_c_predict = classifier.predict(X_c)

        accuracy = accuracy_score(y_true=y_c, y_pred=y_c_predict)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_t = th

        accuracy_list.append(accuracy_score(y_true=y_c, y_pred=y_c_predict))
    plt.scatter(thresh_list, accuracy_list)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.show()
    return best_t


def find_best_params(X, y, X_c, y_c, t):
    estimator_list = np.arange(50, 500, 50)
    depth_list = np.arange(1, 11, 1)
    accuracy_list = []

    best_estimator = estimator_list[0]
    best_depth = depth_list[0]
    best_accuracy = 0

    for e in estimator_list:
        for d in depth_list:
            best_forest = RandomForestClassifier(criterion="entropy", n_estimators=e, n_jobs=2, max_depth=d,
                                                 oob_score=True, max_features="auto")
            best_selection = SelectFromModel(estimator=best_forest, threshold=t)
            best_selection.fit(X, y)
            X = best_selection.transform(X)

            best_forest.fit(X, y)

            X_c = best_selection.transform(X_c)
            y_c_predict = best_forest.predict(X_c)

            accuracy = accuracy_score(y_true=y_c, y_pred=y_c_predict)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_estimator = e
                best_depth = d

            accuracy_list.append(accuracy_score(y_true=y_c, y_pred=y_c_predict))
    return best_estimator, best_depth


def feature_importance(X, y, classifier):
    classifier.fit(X, y)

    # Assess feature importance
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature importance")
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, list(X)[indices[f]], importances[indices[f]]))


def most_frequent_string(Series):
    features = {}
    for element in Series:
        if element is not np.nan and element is not "n" and element is not "nan":
            if element not in features:
                features[str(element)] = 0
                features[str(element)] += 1
            else:
                features[str(element)] += 1
    max_value = max(features.values())
    return [k for k, v in features.items() if v == max_value]

''' Pre-process '''
# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Extract targets and features from train data. Dropped meaningless data: "Name" and "Ticket"
y_train = train_data.get("Survived")
X_train = train_data.iloc[:, 2:].drop(labels="Name", axis=1).drop(labels="Ticket", axis=1)

X_test = test_data.iloc[:, 1:].drop(labels="Name", axis=1).drop(labels="Ticket", axis=1)

# Prepare submission file based on test file
subfile = pd.DataFrame(test_data.iloc[:, 0])
subfile.set_index("PassengerId", inplace=True)

# Show which features have missing data values and how many
print("Features that have missing values: ")
print(X_train.isnull().sum())

# How I will handle the features in X_train that contain missing values:
# Age: replace missing age features with the mean age.
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
imr.fit(X_train[["Age"]])
X_train["Age"] = imr.transform(X_train[["Age"]])
X_train["Age"] = X_train["Age"].astype(str).apply(lambda x: x[:4])

X_train["Cabin"] = X_train["Cabin"].str.replace("\d+", '')
X_train["Cabin"] = X_train["Cabin"].astype(str).apply(lambda x: x[:1])

# Sklearn imputer doesn't allow for string data, so I'll have to find most frequent cabin myself
most_freq_cabin = most_frequent_string(X_train["Cabin"])
X_train["Cabin"] = X_train["Cabin"].str.replace("n", most_freq_cabin[0])

# Same method for embarked feature as for cabin data, as embarked feature is also a string
most_freq_embark = most_frequent_string(X_train["Embarked"])
X_train["Embarked"] = X_train["Embarked"].replace(np.nan, most_freq_embark[0])

# Reveal if any features still have missing values
print("Features that have missing values: ")
print(X_train.isnull().sum())

# Encode nominal data in X_train (sex, embarked, and cabin) using one hot encoder
X_train = pd.get_dummies(X_train, columns=["Sex"])
X_train = pd.get_dummies(X_train, columns=["Embarked"])
X_train = pd.get_dummies(X_train, columns=["Cabin"])

# Drop Cabin_T
X_train = X_train.drop(labels=["Cabin_T"], axis=1)

# And now we do the same thing for our test data
imr.fit(X_test[["Age"]])
X_test["Age"] = imr.transform(X_test[["Age"]])
X_test["Age"] = X_test["Age"].astype(str).apply(lambda x: x[:4])

# We have one null value in X_test fare, so we will take care of that with the imputer also
imr.fit(X_test[["Fare"]])
X_test["Fare"] = imr.transform(X_test[["Fare"]])

X_test["Cabin"] = X_test["Cabin"].str.replace("\d+", '')
X_test["Cabin"] = X_test["Cabin"].astype(str).apply(lambda x: x[:1])

most_freq_cabin = most_frequent_string(X_test["Cabin"])
X_test["Cabin"] = X_test["Cabin"].str.replace("n", most_freq_cabin[0])

most_freq_embark = most_frequent_string(X_test["Embarked"])
X_test["Embarked"] = X_test["Embarked"].replace(np.nan, most_freq_embark[0])

X_test = pd.get_dummies(X_test, columns=["Sex"])
X_test = pd.get_dummies(X_test, columns=["Embarked"])
X_test = pd.get_dummies(X_test, columns=["Cabin"])

''' Algorithm '''
# Find best parameters through this computationally expensive monster that I plan on only using once before commenting
# out
# best_e, best_d = find_best_params(X_train_split, y_train_split, X_cv, y_cv, t)
# print("Best Estimator: ", best_e)
# print("Best depth: ", best_d)

forest = RandomForestClassifier(criterion="gini", n_estimators=350, n_jobs=2, max_depth=7, oob_score=True,
                                max_features="auto")

# Reveal order of feature importance
feature_importance(X_train, y_train, forest)

# Reveal best threshold of features
# t = find_best_thresh(X_train_split, y_train_split, X_cv, y_cv, forest)
# print("Best thresh: ", t)
# Best threshold was found to be 0.1

selection = SelectFromModel(estimator=forest, threshold=0.01)
selection.fit(X_train, y_train)
X_train_selected = selection.transform(X_train)

forest.fit(X_train_selected, y_train)

''' Cross-validation '''
scores = cross_val_score(forest, X_train_selected, y_train, cv=5, scoring='f1_macro')
print("Accuracy: ", scores.mean())

''' Predict '''
X_selected_test = selection.transform(X_test)
y_predict = pd.Series(forest.predict(X_selected_test), index=subfile.index)

subfile["Survived"] = y_predict
subfile.to_csv("submission")

