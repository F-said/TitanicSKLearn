from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def find_best_thresh(X, y, X_c, y_c, estimators, depth):
    thresh_list = np.arange(0, 1, 0.05)
    accuracy_list = []
    best_accuracy = 0
    best_t = thresh_list[0]
    thresh_forest = RandomForestClassifier(criterion="entropy", n_estimators=estimators, n_jobs=2, max_depth=depth,
                                           oob_score=True)
    for th in thresh_list:
        thresh_selection = SelectFromModel(estimator=thresh_forest, threshold=th)
        thresh_selection.fit(X, y)
        X = thresh_selection.transform(X)

        thresh_forest.fit(X, y)

        X_c = thresh_selection.transform(X_c)
        y_c_predict = thresh_forest.predict(X_c)

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
                                                 oob_score=True)
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


def feature_importance(X, y):
    forest = RandomForestClassifier(criterion="entropy", n_estimators=50, n_jobs=2, max_depth=5, oob_score=True)
    forest.fit(X, y)

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

''' Pre-process phase '''
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

# Encode nominal data in X_train (sex, embarked) using ONE HOT ENCODER
X_train = pd.get_dummies(X_train, columns=["Sex"])
X_train = pd.get_dummies(X_train, columns=["Embarked"])

# Encode cabin as ordinal data
size_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'T': 1}
X_train["Cabin"] = X_train["Cabin"].map(size_mapping)

### And now we do the same thing for our test data ###
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

X_test["Cabin"] = X_test["Cabin"].map(size_mapping)

# Reveal order of feature importance
feature_importance(X_train, y_train)

# Create cross validation test set
# Now that I've learned all I could from cv, remove it to give train more data
# X_train_split, X_cv, y_train_split, y_cv = train_test_split(X_train, y_train, test_size=0.25)

''' Algorithm phase '''
# Find best parameters through this computationally expensive monster that I plan on only using once before commenting
# out
# best_e, best_d = find_best_params(X_train_split, y_train_split, X_cv, y_cv, t)
# print("Best Estimator: ", best_e)
# print("Best depth: ", best_d)
# Best n = 50, best depth = 5

# Only process features that are in top 5 relative importance
forest = RandomForestClassifier(criterion="entropy", n_estimators=50, n_jobs=2, max_depth=5, oob_score=True)

# Reveal best threshold of features
# t = find_best_thresh(X_train_split, y_train_split, X_cv, y_cv, 50, 5)
# print("Best thresh: ", t)
# Best threshold was found to be 0

selection = SelectFromModel(estimator=forest, threshold=0.1)
selection.fit(X_train, y_train)
X_train_selected = selection.transform(X_train)

forest.fit(X_train_selected, y_train)

''' Cross-validation phase '''
# X_cv_selected = selection.transform(X_cv)
# y_cv_predict = forest.predict(X_cv_selected)

# print("Accuracy score: ", accuracy_score(y_true=y_cv, y_pred=y_cv_predict))

''' Predict phase '''
X_selected_test = selection.transform(X_test)
y_predict = pd.Series(forest.predict(X_selected_test), index=subfile.index)

subfile["Survived"] = y_predict
subfile.to_csv("submission")

