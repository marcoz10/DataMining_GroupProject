# KNN Classifier

# Import for this project file
from random import randrange

import pandas as pd
import numpy as np
import math
import operator
from sklearn.preprocessing import LabelEncoder

'''
calculate_euclidean_distance(dataset1, dataset2, len): this function will take in 
a set of data that will contain values that we will use to calculate the euclidean 
distance. 
'''
def calculate_euclidean_distance(dataset1, dataset2, len):
    euclidean_distance = 0
    for index in range(len):
        # Calculates the Euclidean Distance.
        # (X_1  - Y_1)^2 + (X_2 - Y_2) + .... (X_n - Y_n)
        euclidean_distance += np.square(dataset1[index] - dataset2[index])
    # We then return the SQRT of these calculation
    return np.sqrt(euclidean_distance)
# End of calculate_euclidean_distance

'''
find_string_columns(dataframe): this function will run through a given dataframe
a pick out the columns that have string data. If it finds a column that does contain 
data the is in string form it will add it's column index/name(s) to a list. Once the function finds all the 
columns that contain strings it returns the list of index/names(s). 
'''
def find_string_columns(dataframe):
    string_columns = []
    column_names = dataframe.columns.values.tolist()
    for col in range(len(dataframe.columns)-1):
        if type(dataframe.iloc[0,col]) is not int or type(dataframe.iloc[0,col]) is not float:
            string_columns.append(column_names[col])

    return string_columns
# End of function find_string_columns


'''
encode_string_columns(dataframe): this function will take dataframe that might contain columns that 
contain string as data type and will convert it to a numerical categorical data. This is done
using a Label Encoder that is include in the sklearn library. 
'''
def encode_string_columns(dataframe, user_test):
    # Create Label Encoder
    labelencoder = LabelEncoder()

    # Call find_string_columns. Returns list of columns with string in them
    list_of_string_cols = find_string_columns(dataframe)
    # Check if we were given test data the needs to encode

    column_names = dataframe.columns.values.tolist()
    frame = pd.DataFrame(user_test, columns=column_names)
    # print(frame)
    dataframe = dataframe.append(frame, ignore_index=True)

    # Use the list and loop through the columns and encode them
    for col in list_of_string_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
        dataframe[col] = dataframe[col].astype(int)
    new_user_test = []
    new_user_test.append(dataframe.iloc[-1, :-1].to_list())
    dataframe = dataframe.iloc[:-1, :]
    return dataframe, new_user_test
    # print(new_test)
    return dataframe, new_test


def knn_classifier(train_data, test_data, k, encode_data = False):

    all_distances = {} # Emypy dictionary to store the euclidean distances of the data

    # Check if data needs to encode
    if encode_data is True:
        old_test_data = test_data
        train_data, test_data = encode_string_columns(train_data, test_data)

    test_data = pd.DataFrame([test_data])
    length_test = test_data.shape[1]

    # Calculate the Euclidean Distance between each row of
    # train_data and test_data
    for x in range(length_test):
        curr_dist = calculate_euclidean_distance(test_data, train_data.iloc[x], length_test)
        all_distances[x] = curr_dist[0]

    # Sort the calculated distances
    sorted_d = sorted(all_distances.items(), key=operator.itemgetter(1))

    # Empty list to hold neighbors
    neighbors = []

    # Pick out the the first k distances in sorted_dist
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    frequent_classes = {}
    # Calculate the most frequent class
    for x in range(len(neighbors)):
        freq= train_data.iloc[neighbors[x]][-1]

        if freq in frequent_classes:
            frequent_classes[freq] += 1
        else:
            frequent_classes[freq] = 1

    sorted_classes = sorted(frequent_classes.items(), key=operator.itemgetter(1), reverse=True)
    return (sorted_classes[0][0], neighbors)
# End of knn_classifier

def evaluate_knn(dataset, algorithm, k, n_folds, encode = False):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_cpy = list(row)
            test_set.append(row_cpy)
            row_cpy[-1] = None
        column_names = dataset.columns.values.tolist()
        frame = pd.DataFrame(train_set, columns=column_names)
        if encode == True:
            predicted = algorithm(frame, test_set,k, True)
        else:
            predicted = algorithm(frame, test_set,k)
        actual = [row[-1] for row in fold]
        accuracy = calculate_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores

def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100

def cross_validation_split(dataset, n_folds):
    data_split = []
    data_copy = dataset.values.tolist()
    fold_size = int(len(dataset) / n_folds)
    for f in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            ind = randrange(len(data_copy))
            fold.append(data_copy.pop(ind))
        data_split.append(fold)
    return data_split


def main(): # TODO: Pick up testing here.
    test = ['purple','SMALL','STRETCH','CHILD']
    data = pd.read_csv('data/balloons.csv')
    k = 3
    result, neighbors = knn_classifer(data, test, k, encode_data=True)

    # test
    print('\nTest = ' , test)
    # Number of K
    print('\nK =', k)
    # Predicted class
    print('\nPredicted Class of the datapoint = ', result)
    # Nearest neighbor
    print('\nNearest Neighbour of the datapoints = ', neighbors)

    test = [34,1,3,100,202,0,0,174,0,0,2,0,2]
    data = pd.read_csv('data/heart.csv')
    k = 2
    result, neighbors = knn_classifer(data, test, k)
    # test
    print('\nTest = ', test)
    # Number of k
    print('\nK = ', k)
    # Predicted class
    print('\nPredicted Class of the datapoint = ', result)
    # Nearest neighbor
    print('\nNearest Neighbour of the datapoints = ', neighbors)


if __name__ == "__main__":
    main()