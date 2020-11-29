# Decision Tree Classifier

# Import for this project file
import pandas as pd
import numpy as np

'''
entropy(target_col): this function will take in a column of data and 
the entropy for it will be calculated.
'''
def entropy(target_col):
    # Identify the unique elements and their associative counts
    elements, counts = np.unique(target_col, return_counts=True)
    # Calculate the entropy associated with the unique elements of the feature and the counts there in
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

'''
info_gain(data,split_attribute_name,target_name): this function will take in 
a data set, the attribute name to be split and the name of the class id or 
target.  It will then calculate the total entropy and then calculate the 
associative entropy for each attribute. Once that is done, the final 
Information Gain is calculated.
'''
def info_gain(data, split_attribute_name, target_name="target"):
    # Calculate the total entropy
    total_entropy = entropy(data[target_name])
    # Identify the unique elements for the attribute we're splitting on
    vals,counts = np.unique(data[split_attribute_name],return_counts=True)
    # Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    # Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


'''
create_decision_tree(data,originaldata,features,target_attribute_name,parent_node_class): 
this function is responsible for creating the dictionary based decision tree.  
It consumes the data, original data, the feature names, classid and the parent node.
This function will recurse over subsets of the original data and build sub trees 
where appropriate.
'''
def create_decision_tree(data,originaldata,features,target_attribute_name="target",parent_node_class=None):
    # For ease of use we are using global variables
    global clf_dt, X_train, y_train
    # Check for purity if it is 100% pure return the value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.ndarray.item(np.unique(data[target_attribute_name]))
    # Check the length of the dataset 
    elif len(data) == 0:
        return np.unique(data[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else: 
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
    
    item_values = [info_gain(data,feature,data.columns[-1]) for feature in features]
    best_feature_index = np.argmax(item_values)
    best_feature = features[best_feature_index]  
    
    # Create a shell for the tree
    tree = {best_feature:{}}
    
    # Make a list of all of the rest of the features
    features = [i for i in features if i!= best_feature]
    
    for value in np.unique(data[best_feature]):
        value = value
        sub_data = data.where(data[best_feature] == value).dropna()
        # Recurse to create subtrees
        subtree = create_decision_tree(sub_data,data,features,target_attribute_name,parent_node_class)
        tree[best_feature][value] = subtree
    
    return tree

'''
predict(query,tree,default): this function is responsible for performing 
the calculation of the provided query for the submitted tree.  The function 
loops over each of the keys in the dictionary and recurses until the 
result is ascertained.
'''
def predict(query,tree,default=1):
    # iterate through the keys 
    for key in list(query.keys()):
        # if you find a key in the list of trees proceed
        if key in list(tree.keys()):
            try:
                # grab the result
                result = tree[key][query[key]]
            except:
                return default
            #result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query,result)
            else:
                return result

'''
test_accuracy(data,tree): this function is responsible for performing 
the accuracy calculation for the decision tree.  It converts the data 
to dictionary, creates a data frame for the predicted values and then 
compares the two results and returns the accuracy of the predictions.
'''
def test_accuracy(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["target"])
    for i in range(len(data)):
        predicted.loc[i,"target"] = predict(queries[i],tree,1.0)
    a = pd.Series(predicted['target'].values)
    b = pd.Series(data['target'].values)
    print("The prediction accuracy is: ", (np.sum(a==b))/len(data)*100,'%')