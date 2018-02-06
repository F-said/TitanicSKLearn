from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd


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

# Extract targets and features from train data. Dropped meaningless data: "Name" and "Ticket". Drop null string data:
# "Embarked" and "Cabin".
y_train = train_data.get("Survived")
X_train = train_data.iloc[:, 2:].drop(labels="Name", axis=1).drop(labels="Ticket", axis=1). \
        drop(labels="Embarked", axis=1).drop(labels="Cabin", axis=1)

X_test = test_data.iloc[:, 1:].drop(labels="Name", axis=1).drop(labels="Ticket", axis=1). \
        drop(labels="Embarked", axis=1).drop(labels="Cabin", axis=1)

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

'''
X_train["Cabin"] = X_train["Cabin"].str.replace("\d+", '')
X_train["Cabin"] = X_train["Cabin"].astype(str).apply(lambda x: x[:1])

# Sklearn imputer doesn't allow for string data (and for good reason), so I'll have to find most frequent cabin myself
most_freq_cabin = most_frequent_string(X_train["Cabin"])
X_train["Cabin"] = X_train["Cabin"].str.replace("n", most_freq_cabin[0])

# Same method for embarked feature as for cabin data, as embarked feature is also a string
most_freq_embark = most_frequent_string(X_train["Embarked"])
X_train["Embarked"] = X_train["Embarked"].replace(np.nan, most_freq_embark[0])
'''

# Reveal if any features still have missing values
print("Features that have missing values: ")
print(X_train.isnull().sum())

# Encode nominal data in X_train (sex, cabin, embarked) using ONE HOT ENCODER
X_train = pd.get_dummies(X_train, columns=["Sex"])
# X_train = pd.get_dummies(X_train, columns=["Cabin"])
# X_train = pd.get_dummies(X_train, columns=["Embarked"])

### And now we do the same thing for our test data ###
imr.fit(X_test[["Age"]])
X_test["Age"] = imr.transform(X_test[["Age"]])
X_test["Age"] = X_test["Age"].astype(str).apply(lambda x: x[:4])

# We have one null value in X_test fare, so we will take care of that with the imputer also
imr.fit(X_test[["Fare"]])
X_test["Fare"] = imr.transform(X_test[["Fare"]])

'''
X_test["Cabin"] = X_test["Cabin"].str.replace("\d+", '')
X_test["Cabin"] = X_test["Cabin"].astype(str).apply(lambda x: x[:1])

most_freq_cabin = most_frequent_string(X_test["Cabin"])
X_test["Cabin"] = X_test["Cabin"].str.replace("n", most_freq_cabin[0])

most_freq_embark = most_frequent_string(X_test["Embarked"])
X_test["Embarked"] = X_test["Embarked"].replace(np.nan, most_freq_embark[0])
'''

X_test = pd.get_dummies(X_test, columns=["Sex"])
# X_test = pd.get_dummies(X_test, columns=["Cabin"])
# X_test = pd.get_dummies(X_test, columns=["Embarked"])

# Add Cabin_T feature to X_test, since no one in X_test was part of the elusive "Cabin_T" class
# X_test.insert(loc=14, column="Cabin_T", value=0)

# Reveal order of feature importance
feature_importance(X_train, y_train)

''' Algorithm phase '''
# Only process features that are in top 5 relative importance
forest = RandomForestClassifier(criterion="entropy", n_estimators=50, n_jobs=2, max_depth=5, oob_score=True)

selection = SelectFromModel(estimator=forest)
selection.fit(X_train, y_train)
X_selected = selection.transform(X_train)

forest.fit(X_selected, y_train)

''' Predict phase '''
X_selected_test = selection.transform(X_test)
y_predict = pd.Series(forest.predict(X_selected_test), index=subfile.index)

subfile["Survived"] = y_predict
subfile.to_csv("submission")


