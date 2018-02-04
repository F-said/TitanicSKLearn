from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

''' Pre-process phase '''
# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Extract targets and features from train data
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
# Cabin: Replace all cabin features with only their respective cabin letters, then replace empty
# cabin features with most frequent cabin
# Embarked: Replace with most frequent embark feature
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
imr.fit(X_train[["Age"]])
X_train["Age"] = imr.transform(X_train[["Age"]])
X_train["Age"] = X_train["Age"].astype(str).apply(lambda x: x[:4])

X_train["Cabin"] = X_train["Cabin"].str.replace("\d+", '')
X_train["Cabin"] = X_train["Cabin"].astype(str).apply(lambda x: x[:1])

### MAKE THIS A FUNCTION ###
# Sklearn imputer doesn't allow for string data (and for good reason), so I'll have to find most frequent cabin myself
titanic_cabin_levels_train = {}
for cabin in X_train["Cabin"]:
    if cabin is not 'n':
        if cabin not in titanic_cabin_levels_train:
            titanic_cabin_levels_train[str(cabin)] = 0
            titanic_cabin_levels_train[str(cabin)] += 1
        else:
            titanic_cabin_levels_train[str(cabin)] += 1
max_value = max(titanic_cabin_levels_train.values())
most_freq_cabin = [k for k, v in titanic_cabin_levels_train.items() if v == max_value]
X_train["Cabin"] = X_train["Cabin"].str.replace("n", most_freq_cabin[0])

### MAKE THIS A FUNCTION ###
# Same method for embarked feature as for cabin data, as embarked feature is also a string
titanic_embark_train = {}
for e in X_train["Embarked"]:
    if e is not np.nan:
        if e not in titanic_embark_train:
            titanic_embark_train[str(e)] = 0
            titanic_embark_train[str(e)] += 1
        else:
            titanic_embark_train[str(e)] += 1
max_value = max(titanic_embark_train.values())
most_freq_embark = [k for k, v in titanic_embark_train.items() if v == max_value]
X_train["Embarked"] = X_train["Embarked"].replace(np.nan, most_freq_embark[0])

# Reveal if any features still have missing values
print("Features that have missing values: ")
print(X_train.isnull().sum())

# Encode nominal data in X_train (sex, cabin, embarked) using ONE HOT ENCODER
X_train = pd.get_dummies(X_train, columns=["Sex"])
X_train = pd.get_dummies(X_train, columns=["Cabin"])
X_train = pd.get_dummies(X_train, columns=["Embarked"])

### And now we do the same thing for our test data ###
imr.fit(X_test[["Age"]])
X_test["Age"] = imr.transform(X_test[["Age"]])
X_test["Age"] = X_test["Age"].astype(str).apply(lambda x: x[:4])

# We have one null value in X_test fare, so we will take care of that with the imputer also
imr.fit(X_test[["Fare"]])
X_test["Fare"] = imr.transform(X_test[["Fare"]])

X_test["Cabin"] = X_test["Cabin"].str.replace("\d+", '')
X_test["Cabin"] = X_test["Cabin"].astype(str).apply(lambda x: x[:1])

titanic_cabin_levels_test = {}
for cabin in X_test["Cabin"]:
    if cabin is not 'n':
        if cabin not in titanic_cabin_levels_test:
            titanic_cabin_levels_test[str(cabin)] = 0
            titanic_cabin_levels_test[str(cabin)] += 1
        else:
            titanic_cabin_levels_test[str(cabin)] += 1
max_value = max(titanic_cabin_levels_test.values())
most_freq_cabin = [k for k, v in titanic_cabin_levels_test.items() if v == max_value]
X_test["Cabin"] = X_test["Cabin"].str.replace("n", most_freq_cabin[0])

titanic_embark_test = {}
for e in X_test["Embarked"]:
    if e is not np.nan:
        if e not in titanic_embark_test:
            titanic_embark_test[str(e)] = 0
            titanic_embark_test[str(e)] += 1
        else:
            titanic_embark_test[str(e)] += 1
max_value = max(titanic_embark_test.values())
most_freq_embark = [k for k, v in titanic_embark_test.items() if v == max_value]
X_test["Embarked"] = X_test["Embarked"].replace(np.nan, most_freq_embark[0])

X_test = pd.get_dummies(X_test, columns=["Sex"])
X_test = pd.get_dummies(X_test, columns=["Cabin"])
X_test = pd.get_dummies(X_test, columns=["Embarked"])

# Add Cabin_T feature to X_test, since no one in X_test was part of the elusive "Cabin_T" class
X_test.insert(loc=14, column="Cabin_T", value=0)

''' Algorithm phase '''
forest = RandomForestClassifier(criterion="entropy", n_estimators=1, n_jobs=2)
forest.fit(X_train, y_train)

''' Predict phase '''
y_predict = pd.Series(forest.predict(X_test), index=subfile.index)

subfile["Survived"] = y_predict
subfile.to_csv("submission")


