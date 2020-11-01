# Python implementation of menu driven 
# Phone Book Directory 
# -------------------------------
# Imports 
import pandas as pd
import os
from os import path
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
import pydot
from PIL import Image

# Global Variables
df = pd.DataFrame()
X_train  = pd.DataFrame() 
X_test = pd.DataFrame() 
y_train = pd.DataFrame() 
y_test = pd.DataFrame() 
graph = pydot.Dot(graph_type='graph')
#-------------------------------

# Decision Tree ----------------------------------------------
# Function used to calculate the entropy of a feature
def entropy(target_col):
    # Identify the unique elements and their associative counts
    elements, counts = np.unique(target_col, return_counts=True)
    # Calculate the entropy associated with the unique elements of the feature and the counts there in
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy
    
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

def create_decision_tree(data,originaldata,features,target_attribute_name="target",parent_node_class=None):
    # Check for purity if it is 100% pure return the value
    if len(np.unique(data[target_attribute_name])) <= 1:
        #return np.unique(data[target_attribute_name])
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
    tree = {best_feature:{}}
    
    features = [i for i in features if i!= best_feature]
    
    for value in np.unique(data[best_feature]):
        value = value
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = create_decision_tree(sub_data,data,features,target_attribute_name,parent_node_class)
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

def draw(parent_name,child_name):
    global graph
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


def test(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])
    #for i in range(len(data)):
        #predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0)
    #print("Hello")
    #print("The prediction accuracy is: ", (np.sum(predicted["predicted"])==data["target"])/len(data)*100,'%')
    get_prediction = {'age':'senior','credit_rating':'fair'}
    print(predict(get_prediction,tree))    
# -------------------------------------------------------------

# Function to build the decision tree
def decision_tree():
    # Call tree build
    tree = create_decision_tree(X_train,X_train,X_train.iloc[:,:-1].columns,y_train.columns[0])
    
    # Did our tree return with data?
    if(not bool(tree)):
        print("The tree did not load properly!")
        exit()
    else:
        # We've got a tree, let's offer the user some options
        print("Your decision tree has loaded!")
        print("Please choose any choice from below -\n") 
        print("(1) View Graphical Tree") 
        print("(2) View Text Tree") 
        print("(3) Return to Main")  
        choice = int(input())
        if(choice == 1):
            # Graphical display of the decision tree
            global graph
            graph = pydot.Dot(graph_type='graph')
            visit(tree)
            graph.write_png('cs4373_decision_tree.png')
            img = Image.open('cs4373_decision_tree.png')
            img.show()
        elif(choice == 2):
            # Text based display of the decision tree
            pprint(tree)
        elif(choice == 3):
            # Proceed back to the main menu
            main()
        else:
            main()

    test(X_test,graph)

    # Give the user an opp to continue with the application
    print("Do you want to perform more operations? (y / n)")
    
    # Act on the users input
    choice = input().strip()
    if choice == "y":
        main()

    
# Function to load a global data frame 
def load_data():
    # Give instructions to the user 
    print("\n\nInput a csv file to load, please include file name and .csv extension")
    
    # Get filename from the user
    file_name = input().strip()
    
    # Let's go ahead and for the filename to be a csv 
    file_name = path.splitext(file_name)[0] + ".csv"
    
    # check to make sure the file exists
    if(path.exists(file_name)):
            df = pd.read_csv(file_name)
    else:
        # If the file does not exist we need to give the user an opp 
        # to supply a new file
        print("The provided filename (" + file_name + ") does not exist!")
        print("\nWould you like to try a different file? (y / n)")
        choice = input().strip()
        if choice == "y":
            load_data()
        else:
            exit()
    
    # Give the user an opp to see the contents of the file they loaded
    print("\nWould you like to see the contents of your file?  (y / n)")
    choice = input().strip()
    if choice == "y":
        print(df)
    
    global X_train, X_test, y_train, y_test
    
    # We want to create both training and testing datasets, we will use test_train_split to do this
    X_train, X_test, y_train, y_test = train_test_split(df, df.iloc[:,-1:], test_size=0.30, random_state=42)
    
    # Give the user an opp to continue with the application
    print("Do you want to perform more operations? (y / n)")
    
    # Act on the users input
    choice = input().strip()
    if choice == "y":
        main()


# Main Function for Menu-Driven 
def main(): 
	print("Please choose any choice from below -\n") 
	print("(1) Load Dataset") 
	print("(2) Decision Tree Induction") 

	choice = int(input()) 

	choice_dict = { 
		1: load_data, 
		2: decision_tree, 
	} 

	choice_dict[choice]() 

os.system('cls')
if __name__ == "__main__": 
	print("---------------------- CS4373 Class Project - Classifiers ----------------------") 

main() 
