# Python implementation of menu driven 
# Phone Book Directory 
# -------------------------------
# Imports 
import pandas as pd
import os
from os import path
import numpy as np
from random import seed
from random import randrange
from pprint import pprint
from sklearn.model_selection import train_test_split
import pydot
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
        return dataset_split

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = DTC.test_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores

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
    user_test = [item for item in input('Enter list for test: ').split(',')]
    data = [user_test]
    cols = X_train.columns.tolist()
    data = pd.DataFrame(data,columns=cols[:-1])
    queries = data.to_dict(orient="records")
    result = DTC.predict(queries[0],tree)
    print(result)

#******************************************************
# Menu Item for Decision Tree Menu
def built_in_dtclf():
    global X_train, X_test, clf_dt

    # Since the built in decision tree requires numeric values, let's
    # encode the categorical variables.
    df_train = X_train.copy()
    df_test = X_train.copy()
    for cols in df_train.columns[:-1]:
        df_train[cols] = pd.Categorical(df_train[cols]).codes
        df_test[cols] = pd.Categorical(df_test[cols]).codes

    # Let's copy the classifier 
    y_train = df_train['target'].copy()
    y_test = df_test['target'].copy()

    # Let's remove the classifier from the dataset 
    df_train = df_train.drop(['target'],axis=1)
    df_test = df_test.drop(['target'],axis=1)
    clf_dt = clf_dt.fit(df_train,y_train)
    y_pred = clf_dt.predict(df_test)
    score = accuracy_score(y_test, y_pred)
    print("The built-in accuracy is: ", score*100,'%')

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
    chk_if_cat = input('Does this data contain categorical data? (y/n)\n').upper()
    if chk_if_cat == 'Y':
        try:
            model, foo = nb.summarize_by_class(X_train, X_test.iloc[0, :-1])
            accuracy_nb = nb.get_accuracy(model, X_test, y_test, True)
            print('\nAccuracy of Naive Bayes Model = %s\n' % (accuracy_nb/100.00))
            # Use the built in functions
            df_train = X_train.copy()
            df_test = X_test.copy()
            # encode all columns
            for cols in df_train.columns[:-1]:
                df_train[cols] = pd.Categorical(df_train[cols]).codes
                df_test[cols] = pd.Categorical(df_test[cols]).codes
                # Get y_(train/test)
            y_train_pred = df_train['target'].copy()
            y_test_pred = df_test['target'].copy()
            # Take out classifer from the data set
            df_train = df_train.drop(['target'], axis=1)
            df_test = df_test.drop(['target'], axis=1)
            # Make classifier
            clf = GaussianNB()
            clf.fit(df_train, y_train.to_numpy().ravel())
            # get score
            score = clf.score(df_test, y_test.to_numpy().ravel())
            print('The built-in accuracy is = %s\n' % score)
            # Print out which classifier is better
            if (float(accuracy_nb)/100.00) > score:
                print('The accuracy of our model was ' + str(((float(accuracy_nb)/100.00 - score) * 100.00)) +
                      '% better then the built-in model.\n')
            elif (float(accuracy_nb)/100.00) == score:
                print('The accuracy of our model and the built-in model are the same\n')
            else:
                print('The accuracy of the built-in model was '+ str(((score - float(accuracy_nb)/100.00) * 100.00)) +
                      '% better then our model.\n')
        except:
            print('Error: Problems with calculation\n')
    else:
        try:
            model = nb.summarize_by_class(X_train)
            accuracy_nb = nb.get_accuracy(model, X_test, y_test)
            print('\nAccuracy of Naive Bayes Model = %s\n' % (accuracy_nb/100.00))
            # Built in
            clf = GaussianNB()
            clf.fit(X_train, y_train.to_numpy().ravel())
            score = clf.score(X_test, y_test.to_numpy().ravel())
            print('The built-in accuracy is = %s\n' % score)
            # Print out which classifier is better
            if (float(accuracy_nb) / 100.00) > score:
                print('The accuracy of our model was ' + str(((float(accuracy_nb) / 100.00 - score) * 100.00)) +
                      '% better then the built-in model.\n')
            elif (float(accuracy_nb) / 100.00) == score:
                print('The accuracy of our model and the built-in model are the same\n')
            else:
                print(
                    'The accuracy of the built-in model was ' + str(((score - float(accuracy_nb) / 100.00) * 100.00)) +
                    '% better then our model.\n')
        except:
            print('Error: Problems with calculation\n')
# End calculate_accuracy_bayes

#******************************************************
def make_prediction():
    # Ask for a list to test with the classifier
    print('Please enter the test list to classify.')
    user_test = [item for item in input('Enter list for test: ').upper().split(',')]

    # Ask user if data set contains categorical data
    categorical_or_not = input('\nDoes this data set have categorical data? (y/n)\n').upper()

    # Train the classifier with training data
    if (categorical_or_not == 'Y'):
        try:
            model, user_test_cat = nb.summarize_by_class(X_train, user_test)
            # Make prediction with our model
            label, probs = nb.give_predict(model, user_test_cat[:-1])
            print('Data=%s, Predicted: %s\n' % (user_test, label))
        except:
            print('Error: Problem with calculations.')
    else:
        try:
            user_test_int = list(map(float, user_test))
            model = nb.summarize_by_class(X_train)
            label, probs = nb.give_predict(model, user_test_int)
            print('Data=%s, Predicted: %s\n' % (user_test, label))
        except:
            print('Error: Problem with calculations.\n')

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
    global y_train_
    encoded = input('Does this data contain categorical values? (y/n)\n').upper()

    if encoded == 'Y':
        try:
            accuracy_knn = kc.get_accuracy_knn(X_train, X_test, y_test, True)
            print('\nAccuracy of KNN Model = %s\n' %(accuracy_knn))
            # Use the built in functions
            df_train = X_train.copy()
            df_test = X_test.copy()
            # encode all columns
            for cols in df_train.columns[:-1]:
                df_train[cols] = pd.Categorical(df_train[cols]).codes
                df_test[cols] = pd.Categorical(df_test[cols]).codes
            # Take out classifer from the data set
            df_train = df_train.drop(['target'], axis=1)
            df_test = df_test.drop(['target'], axis=1)
            # Make classifier
            clf = KNeighborsClassifier()
            clf.fit(df_train, y_train.to_numpy().ravel())
            score = clf.score(df_test, y_test.to_numpy().ravel())
            print('Accuracy of Built-in KNN Model = %s\n' % score)
            # Print out which classifier is better
            if (float(accuracy_knn) / 100.00) > score:
                print('The accuracy of our model was ' + str(((float(accuracy_knn) / 100.00 - score) * 100.00)) +
                      '% better then the built-in model.\n')
            elif (float(accuracy_knn) / 100.00) == score:
                print('The accuracy of our model and the built-in model are the same\n')
            else:
                print(
                    'The accuracy of the built-in model was ' + str(((score - float(accuracy_knn) / 100.00) * 100.00)) +
                    '% better then our model.\n')
        except:
            print('Error: Problem with calculation\n')
    else:
        try:
            accuracy_knn = kc.get_accuracy_knn(X_train, X_test, y_test)
            print('\nAccuracy of KNN Model = %s\n' %
              (accuracy_knn/100.00))
            # Built in
            clf = KNeighborsClassifier()
            clf.fit(X_train, y_train.to_numpy().ravel())
            score = clf.score(X_test,y_test.to_numpy().ravel())
            print('Accuracy of Built-in KNN Model = %s\n' % score)
            if (float(accuracy_knn) / 100.00) > score:
                print('The accuracy of our model was ' + str(((float(accuracy_knn) / 100.00 - score) * 100.00)) +
                      '% better then the built-in model.\n')
            elif (float(accuracy_knn) / 100.00) == score:
                print('The accuracy of our model and the built-in model are the same\n')
            else:
                print('The accuracy of the built-in model was ' + str(((score - float(accuracy_knn) / 100.00) * 100.00)) +
                    '% better then our model.\n')
        except:
            print('Error: Problem with calculation\n')
# ********************************************************
def make_prediction_knn():
    # Ask for a list to test with the classifier
    print('Please enter the test list to classify.')
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
