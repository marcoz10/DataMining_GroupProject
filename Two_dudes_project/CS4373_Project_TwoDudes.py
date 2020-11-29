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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
sys.path.insert(1, '..DataMining_GroupProject/Two_dudes_project')
import Naive_Bayes_Classifier as nb
import knn_classifier as kc
import Decision_Tree_Classifier as DTC

# Global Variables
clf_dt = DecisionTreeClassifier(criterion="entropy")
df = pd.DataFrame()
X_train  = pd.DataFrame() 
X_test = pd.DataFrame() 
y_train = pd.DataFrame() 
y_test = pd.DataFrame() 
graph = pydot.Dot(graph_type='graph')
edges = []
#-------------------------------


def draw(parent_name,child_name):
    global graph
    edge = pydot.Edge(parent_name,child_name)
    if edge.to_string() not in edges:
        edges.append(edge.to_string())
        graph.add_edge(edge)

def visit(node,parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            if parent:
                draw(parent,k)
            visit(v,k)
        else:
            draw(parent,k)
            draw(k, k+'_'+v)


#******************************************************
# Menu Item From Main Menu
def decision_tree():
    # Call tree build
    global tree
    tree = DTC.create_decision_tree(X_train,X_train,X_train.iloc[:,:-1].columns,y_train.columns[0])
    # Did our tree return with data?
    if(not bool(tree)):
        print("The tree did not load properly!")
        exit()
    else:
        # We've got a tree, let's offer the user some options
        print("Your decision tree has loaded!")
    decision_tree_menu()

#******************************************************
# Decision Tree Menu
def decision_tree_menu():
    print("Please choose any choice from below -\n") 
    print("(1) View Graphical Tree") 
    print("(2) View Text Tree") 
    print("(3) Test the Accuracy of the Tree")
    print("(4) Generate a prediction based on a tuple")
    print("(5) Test built in classifier")
    print("(6) Return to Main")
    
    choice = int(input())
    
    choice_dict = {
        1: display_graphical_decision_tree,
        2: display_text_decision_tree, 
        3: perform_test,
        4: user_class_prediction,
        5: built_in_dtclf,
        6: main
	} 
    choice_dict[choice]()

    # Give the user an opp to continue with the application
    print("Do you want to perform more decision tree operations? (y / n)")
    
    # Act on the users input
    choice = input().strip()
    if choice == "y":
        decision_tree_menu()
    else:
        main()

#******************************************************
# Menu Item for Decision Tree Menu
def display_graphical_decision_tree():
    # Graphical display of the decision tree
    global graph,tree
    graph = pydot.Dot(graph_type='graph')
    visit(tree)
    graph.write_png('cs4373_decision_tree.png')
    img = Image.open('cs4373_decision_tree.png')
    img.show()

#******************************************************
# Menu Item for Decision Tree Menu
def display_text_decision_tree():
    global tree
    pprint(tree)

#******************************************************
# Menu Item for Decision Tree Menu
def perform_test():
    global tree
    DTC.test_accuracy(X_test,tree)

#******************************************************
# Menu Item for Decision Tree Menu
def user_class_prediction():
    global tree
    data = [['middle_aged', 'high', 'nope','excellent','yes']]
    data = pd.DataFrame(data,columns=['age', 'income','student','credit_rating','target'])
    queries = data.iloc[:,:-1].to_dict(orient="records")
    result = DTC.predict(queries[0],tree)
    print(result)

#******************************************************
# Menu Item for Decision Tree Menu
def built_in_dtclf():
    global X_train, X_test, clf_dt
    df_train = X_train.copy()
    df_train['age'] = pd.Categorical(df_train['age']).codes
    df_train['income'] = pd.Categorical(df_train['age']).codes
    df_train['student'] = pd.Categorical(df_train['student']).codes
    df_train['credit_rating'] = pd.Categorical(df_train['credit_rating']).codes
    df_test = X_train.copy()
    df_test['age'] = pd.Categorical(df_test['age']).codes
    df_test['income'] = pd.Categorical(df_test['age']).codes
    df_test['student'] = pd.Categorical(df_test['student']).codes
    df_test['credit_rating'] = pd.Categorical(df_test['credit_rating']).codes

    # Let's copy the classifier 
    y_train = df_train['target'].copy()
    y_test = df_test['target'].copy()

    # Let's remove the classifier from the dataset 
    df_train = df_train.drop(['target'],axis=1)
    df_test = df_test.drop(['target'],axis=1)
    clf_dt = clf_dt.fit(df_train,y_train)
    y_pred = clf_dt.predict(df_test)
    score = accuracy_score(y_test, y_pred)
    
    print(score)

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

def close_app():
    quit()

#******************************************************
def calculate_accuracy_bayes():
    print('Does this data contain categorical values?')
    chk_if_cat = input('Y|y for YES :: N|n for NO\n').upper()
    if chk_if_cat == 'Y':
        model, foo = nb.summarize_by_class(X_train, X_test.iloc[0, :-1])
        accuracy_nb = nb.get_accuracy(model, X_test, y_test, True)
    else:
        model = nb.summarize_by_class(X_train)
        accuracy_nb = nb.get_accuracy(model, X_test, y_test)

    print('\nAccuracy of Naive Bayes Model = %s\n' % (accuracy_nb))


# End calculate_accuracy_bayes

#******************************************************
def make_prediction():
    # Ask for a list to test with the classifier
    print('Please enter the test list to classify.')
    print('Pleas enter list seperated by commas e.g. "1,2,3,4,5"\n')
    user_test = [item for item in input('Enter list for test: ').upper().split(',')]

    # Ask user if data set contains categorical data
    categorical_or_not = input('\nDoes this data set have categorical data? (y/n)\n').upper()

    # Train the classifier with training data
    if (categorical_or_not == 'Y'):
        model, user_test_cat = nb.summarize_by_class(X_train, user_test)
        # Make prediction
        label, probs = nb.give_predict(model, user_test_cat[:-1])
        print('Data=%s, Predicted: %s\n' % (user_test, label))
    else:
        user_test_int = list(map(float, user_test))
        model = nb.summarize_by_class(X_train)
        label, probs = nb.give_predict(model, user_test_int)
        print('Data=%s, Predicted: %s\n' % (user_test, label))

    check_probs = input('Would you like to see the probablities for the classifier? (y/n)').upper()
    if (check_probs == 'Y'):
        for x in probs:
            print(str(x) + ' : ' + str(probs[x]) + '\n')
    else:
        # Give the user an opp to continue with the application
        print("Do you want to perform more naive bayes operations? (y / n)")

        # Act on the users input
        choice = input().strip()
        if choice == "y":
            bayes_menu()
        else:
            main()
# End make_prediction
#******************************************************
# Menu for Bayes Classifier
def bayes_menu():
    print('(1) Test Accuracy')
    print('(2) Make prediction')
    print('(3) Return to Main')
    choice = int(input())

    choice_dict = {
        1: calculate_accuracy_bayes,
        2: make_prediction,
        3: main
    }
    choice_dict[choice]()

    # Give the user an opp to continue with the application
    print("Do you want to perform more naive bayes operations? (y / n)")

    # Act on the users input
    choice = input().strip()
    if choice == "y":
        bayes_menu()
    else:
        main()
# End bayes_menu

# ********************************************************
def calculate_accuracy_knn():
    # Ask if data is categorical
    encoded = input('Does this data contain categorical values? (y/n)\n').upper()

    if encoded == 'Y':
        print('\nAccuracy of KNN Model = %s\n' %
              (kc.get_accuracy_knn(X_train, X_test, y_test, True)))
    else:
        print('\nAccuracy of KNN Model = %s\n' %
              (kc.get_accuracy_knn(X_train, X_test, y_test)))

# ********************************************************
def make_prediction_knn():
    # Ask for a list to test with the classifier
    print('Please enter the test list to classify.')
    print('Pleas enter list seperated by commas e.g. "1,2,3,4,5" or "cat, dog, frog"\n')
    user_test = [item for item in input('Enter list for test: ').upper().split(',')]

    # Ask how many clusters for knn to use
    k = int(input('\nHow many clusters would you like:\n'))

    # Ask user if data set contains categorical data
    categorical_or_not = input('\nDoes this data set have categorical data? (y/n)\n').upper()
    # Train the classifier with training data
    if categorical_or_not == 'Y':
        try:
            result, neighbors = kc.knn_classifer(X_train, user_test, k, encode_data=True)
            # test
            print('\nTest = ', user_test)
            # Number of K
            print('\nK =', k)
            # Predicted class
            print('\nPredicted Class of the datapoint = ', result)
            # Nearest neighbor
            print('\nNearest Neighbour of the datapoints = \n', neighbors)
        except:
            print('k clusters out of range\n')
    else:
        try:
            user_test_int = list(map(float, user_test))
            result, neighbors = kc.knn_classifer(X_train, user_test_int, k)
            # test
            print('\nTest = ', user_test)
            # Number of k
            print('\nK = ', k)
            # Predicted class
            print('\nPredicted Class of the datapoint = ', result)
            # Nearest neighbor
            print('\nNearest Neighbour of the datapoints = \n', neighbors)
        except:
            print('k clusters out of range\n')

    # Give the user an opp to continue with the application
    print("Do you want to perform more knn operations? (y / n)")

    # Act on the users input
    choice = input().strip()
    if choice == "y":
        knn_menu()
    else:
        main()
#*********************************************************
def knn_menu():
    print('(1) Test Accuracy')
    print('(2) Make prediction')
    print('(3) Return to Main')
    choice = int(input())

    choice_dict = {
        1: calculate_accuracy_knn,
        2: make_prediction_knn,
        3: main
    }
    choice_dict[choice]()

    # Give the user an opp to continue with the application
    print("Do you want to perform more knn operations? (y / n)")

    # Act on the users input
    choice = input().strip()
    if choice == "y":
        knn_menu()
    else:
        main()
# End knn_menu
# Main Function for Menu-Driven 
def main():
    print("Please choose any choice from below -\n")
    print("(1) Load Dataset")
    print("(2) Decision Tree Induction")
    print('(3) Navie Bayes Classifier')
    print('(4) KNN Classifier')
    print("(5) Exit the Application")
    choice = int(input())
    
    choice_dict = { 
		1: load_data, 
		2: decision_tree,
        3: bayes_menu,
        4: knn_menu,
        5: close_app
	}
    
    choice_dict[choice]() 

os.system('cls')
if __name__ == "__main__": 
	print("---------------------- CS4373 Class Project - Classifiers ----------------------") 

main() 
