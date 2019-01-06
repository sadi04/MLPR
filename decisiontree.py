import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder
import math

column = ['age','workclass','fnlwgt','education','education-num',
            'marital-status', 'occupation','relationship','race','sex',
            'capital-gain','capital-loss', 'hours-per-week', 'native-country','income']
dataframe = pd.read_csv("adult_test.csv", names=column)

dataframe.replace(' ?', np.NaN, inplace=True)
dataframe.drop(columns=['age','fnlwgt','capital-gain','capital-loss', 'hours-per-week'],inplace=True)
dataframe.dropna(inplace=True)


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):

        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

#dataframe = MultiColumnLabelEncoder(columns = ['workclass','education','marital-status', 'occupation',
#                                   'relationship','race','sex','native-country','income']).fit_transform(dataframe)

#print(dataframe)


def entropy(target_col):

    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data, split_attribute_name, target_name="income"):

    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def decisionTree(data, originaldata, features, depth, target_attribute_name="income", parent_node_class = None):

    if depth == 0:
        return np.unique(originaldata[target_attribute_name])[
        np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(data) == 0:
        return parent_node_class
    elif len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(features) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    else:
        depth -= 1
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]  # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = decisionTree(sub_data, data, features, depth, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        for value in np.unique(data[best_feature]):
            if value not in list(tree[best_feature].values()):
                sub_data = pd.DataFrame(columns=column)
                subtree = decisionTree(sub_data, data, features, depth, target_attribute_name, parent_node_class)
                tree[best_feature][value] = subtree

        return (tree)

tree = decisionTree(dataframe,dataframe,dataframe.columns[:-1],10)
#print(tree)

def searchTree(tree, row):
    toSearch = tree

    while type(toSearch) == dict:
        key = list(toSearch.keys())
        #print(toSearch)
        index = row[0].index(key[0])
        val = row[1][index]
        toSearch = toSearch[key[0]]
        toSearch = toSearch[val]
    return toSearch

cor=0
for i in range(len(dataframe)):
    rows = []
    rows.append(list(dataframe))
    va = dataframe.values.tolist()[i]
    rows.append(va)
    #print(rows)
    result=searchTree(tree,rows)
    if result == rows[1][-1]:
        cor+=1
print(cor/len(dataframe))

def normalize(weight):
    tot = sum(weight)
    normal = []
    for i in range(len(weight)):
        normal.append(weight[i]/tot)
    return normal

def adaboost(examples, k):
    h = []
    z = []
    w = []
    for i in range(len(examples)):
        w.append(1.0/len(examples))

    for i in range(k):
        data = examples.sample(n=2000,weights=w)
        newTree = decisionTree(data,data,dataframe.columns[:-1],k)
        error = 0
        for i in range(len(data)):
            rows = []
            rows.append(list(data))
            va = data.values.tolist()[i]
            rows.append(va)
            if searchTree(newTree, rows) != rows[1][-1]:
                error += w[i]
        if (error > 0.5):
            continue
        for i in range(len(data)):
            rows = []
            rows.append(list(data))
            va = data.values.tolist()[i]
            rows.append(va)
            if searchTree(newTree, rows) == rows[1][-1]:
                w[i]*=(error/(1 - error))
        w = normalize(w)
        z.append(math.log10((1 - error) / error))
        h.append(newTree)
        return h,z


testframe = pd.read_csv("adult_train.csv", names=['age','workclass','fnlwgt','education','education-num',
                                                'marital-status', 'occupation','relationship','race','sex',
                                                 'capital-gain','capital-loss', 'hours-per-week', 'native-country','income'])

testframe.replace(' ?', np.NaN, inplace=True)
testframe.drop(columns=['age','fnlwgt','capital-gain','capital-loss', 'hours-per-week'],inplace=True)
testframe.dropna(inplace=True)

treeSpace, factor = adaboost(dataframe, 20)

correct = 0

print(len(treeSpace))
for i in range(len(testframe)):
    decision = 0
    rows = []
    rows.append(list(testframe))
    va = testframe.values.tolist()[i]
    rows.append(va)
    true = 0
    false = 0
    for k in range(len(treeSpace)):
        result=searchTree(treeSpace[k], rows)
        if  result== rows[1][-1]:
            true+=1
            decision += factor[k]
        else:
            false += 1
            decision -= factor[k]
    print(decision, true, false)
    if (decision > 0 and (rows[1][-1] == ' >50K.')) or (decision <= 0 and (rows[1][-1] == ' <=50K.')):
        correct += 1

print("Accuracy in test set : ",(correct*100) / len(testframe),"%")
