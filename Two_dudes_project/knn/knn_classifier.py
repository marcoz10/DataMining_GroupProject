# KNN Classifier

# Import for this project file
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
def encode_string_columns(dataframe, labeled_test):
    # Create Label Encoder
    labelencoder = LabelEncoder()

    # Call find_string_columns. Returns list of columns with string in them
    list_of_string_cols = find_string_columns(dataframe)
    # Check if we were given test data the needs to encode

    labeled_test.append(None) # Adding none to the end to replace target value
    column_names = dataframe.columns.values.tolist()
    new_row = dict(zip(column_names,labeled_test))
    # print(new_row)
    dataframe = dataframe.append(new_row, ignore_index= True)
        # print(dataframe.tail())
    # Use the list and loop through the columns and encode them
    for col in list_of_string_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])

    new_test = dataframe.iloc[-1,:-1].to_numpy()
    dataframe = dataframe.iloc[:-1,:]
    # print(new_test)
    return dataframe, new_test


def knn_classifer(train_data, test_data, k, encode_data = False):
    all_distances = {} # Emypy dictionary to store the euclidean distances of the data


    # Check if data needs to encode
    if encode_data is True:
        old_test_data = test_data
        train_data, test_data = encode_string_columns(train_data, test_data)

    length_test = test_data.shape[1]

    # Calculate the Euclidean Distance between each row of
    # train_data and test_data
    for pnt in range(len(length_test)):
        curr_dist = calculate_euclidean_distance(test_data, train_data.iloc[pnt], length_test)
        all_distances[pnt] = curr_dist

    # Sort the calculated distances
    sorted_d = sorted(all_distances.items(), key=operator.itemgetter())

    # Empty list to hold neighbors
    neighbors = []

    # Pick out the the first k distances in sorted_dist
    for neigh in range(k):
        neighbors.append(sorted_d[neigh][0])
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

def main(): # TODO: Pick up testing here.
    # test = ['purple','SMALL','STRETCH','CHILD']
    # test_df = pd.DataFrame(np.array(test))
    test = [['purple','SMALL','STRETCH','CHILD']]
    test_df = pd.DataFrame(test)
    print(test_df)


if __name__ == "__main__":
    main()