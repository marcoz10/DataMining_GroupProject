# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:14:22 2020

@author: sgarc
"""
#https://www.youtube.com/watch?v=K5QlpAqOtTE&t=6341s
import numpy as np
import pandas as pd
from pprint import pprint
import pydot


def entropy(target_col):
    elements,counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy
    
def InformationGain(data, split_attribute_name, target_name="target"):
    total_entropy = entropy(data[target_name])
    #print(total_entropy)
    vals,counts = np.unique(data[split_attribute_name],return_counts=True)
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    #print(Information_Gain)
    return Information_Gain

def ID3(data,originaldata,features,target_attribute_name="target",parent_node_class=None):
    print(features)
    # Check for purity if it is 100% pure return the value
    if len(np.unique(data[target_attribute_name])) <= 1:
        #return np.unique(data[target_attribute_name])
        return np.asscalar(np.unique(data[target_attribute_name]))
    # Check the length of the dataset 
    elif len(data) == 0:
        return np.unique(data[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else: 
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
    
    item_values = [InformationGain(data,feature,data.columns[-1]) for feature in features]
    #print(item_values)
    best_feature_index = np.argmax(item_values)
    best_feature = features[best_feature_index]
    
    tree = {best_feature:{}}
    
    features = [i for i in features if i!= best_feature]
    
    for value in np.unique(data[best_feature]):
        value = value
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = ID3(sub_data,data,features,target_attribute_name,parent_node_class)
        # I need to find a way to remove the yes/no array entries
        #if(isarray(subtree)): print(subtree)
        tree[best_feature][value] = subtree
    return tree


def predict(query,tree,default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query,result)
            else:
                return result
            

def train_test_split(dataset):
    training_data = dataset.iloc[:14].reset_index(drop=True)
    testing_data = dataset.iloc[14:].reset_index(drop=True)
    return training_data, testing_data

def test(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0)
    
    print("The prediction accuracy is: ", (np.sum(predicted["predicted"])==data["target"])/len(data)*100,'%')
    
    
    

def draw(parent_name,child_name):
    edge = pydot.Edge(parent_name,child_name)
    graph.add_edge(edge)

def visit(node,parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            if parent:
                draw(parent,k)
            visit(v,k)
        else:
            draw(parent,k)
            draw(k,v)


#, names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
dataset = pd.read_csv('test2.csv')


training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]

tree = ID3(training_data,training_data,training_data.columns[:-1])
test(testing_data,tree)
pprint(tree)

graph = pydot.Dot(graph_type='graph')
visit(tree)
graph.write_png('example.png')

get_prediction = {'age':'senior','credit_rating':'fair','student':'nope'}
print(predict(get_prediction,tree))