# Imports for this file
import pandas as pd
from math import sqrt
from math import exp
from math import pi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Functions

'''
separate_by_classes(data): this function will take in a dataset in form of 
a dataframe. With this data it will take the all columns in the range of the 
(0 - n-1) where n is the number of column. Then by using the values of the last 
column it divides the rows of the separated data, and groups them by the value of 
their last column as a category. 
'''
def separate_by_classes(data):
    separated = {} # dictionary to hold the rows for each target
    for row in range(len(data)):
        vector = data.iloc[row, :-1].values.tolist()
        class_value = data.iloc[row, -1]
        if class_value in separated:
            separated[class_value].append(vector)
        else:
            separated[class_value] = []

    return separated

# End of function separate_by_class

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
def encode_string_columns(dataframe,user_test):
    # Create Label Encoder
    labelencoder = LabelEncoder()

    # Call find_string_columns. Returns list of columns with string in them
    list_of_string_cols = find_string_columns(dataframe)
    # Check if we were given test data the needs to encode
    # Adding none to the end to replace target value
    user_test.append(None)
    column_names = dataframe.columns.values.tolist()
    new_row = dict(zip(column_names,user_test))
    # print(new_row)
    dataframe = dataframe.append(new_row, ignore_index= True)
        # print(dataframe.tail())
    # Use the list and loop through the columns and encode them
    for col in list_of_string_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])


    user_test = dataframe.iloc[-1,:-1].to_numpy()
    dataframe = dataframe.iloc[:-1,:]
    # print(new_test)
    return dataframe, user_test

# End of function encode_string_columns
'''
mean(nums): returns the means of a set of numbers
'''
def mean(nums):
    return sum(nums) / float(len(nums))

# End of function mean

'''
stdev(nums): returns the standard deviation of a set of numbers
'''
def stdev(nums):
    avg = mean(nums)
    variance = sum([(x - avg) ** 2 for x in nums]) / float(len(nums) - 1)
    return sqrt(variance)

# End of function
'''
summarize_dataset(dataset): takes a dataset (dataframe) and calculates a
summary list using the functions mean(), stdev(), len() on all the columns
in the dataframe. Then returns this list without the last element because this
contains the target columns summary
'''
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries
# End of function summarize_dataset

'''
summarize_by_class(dataset, labeled_teset=None): this function takes the dataset
 (dataframe) and seperates the rows into different classes that are determined by the
 value of the target value. Then it will run through the classes row by row. In
each row it will use the summarize_dataset to summaries of each column. Will return 
as dictionary of seperated and summarized classes. If the given a labeled_test parameter
it will return a dictionary of seperated and cummarized classes and a encoded test that will 
be used in the give predict function later on. 
'''
def summarize_by_class(dataset, user_test = None ):
    # Check if labeled test is given. If so send it through
    # the encode_string_columns function to encode dataset
    # and test.
    if user_test is not None:
        dataset,new_user_test = encode_string_columns(dataset, user_test)

    separated = separate_by_classes(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)

    # Check if test was given to be encoded. If so return summaries and
    # a new_test. new_test is the same test but encoded.
    if user_test is not None:
        return summaries , new_user_test
    else:
        return summaries

'''
calculate_probaility(x, mean, stdev): this function will calculate the
bayes probability of an event occuring given x, x's mean, and x's stdev 
'''
def calculate_probability(x, mean, stdev):
    if stdev == 0.0 or stdev == 0:
        stdev = 0.0000001
    if mean == 0.0 or mean == 0:
        mean = 0.0000001
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

'''
calculate_class_probabilities(summaries, row): calculates all the probabilities
of every class items in all classes compared to a given row by the user. Then
returns a model for use in a prediction
'''
def calculate_class_probabilities(summaries, test):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        # print('The probability of class %d: %.3f' % (class_value, probabilities[class_value]))
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(test[i], mean, stdev)
    return probabilities

'''
give_predict(summaries, test): this function uses that model that is 
given by the calculate_class_probabilities function and a test given 
by the user. Then compares the test to the probabilities in the model
to deterimine the best prediction for what the test should be.
'''
def give_predict(summaries, test):
    probabilities = calculate_class_probabilities(summaries, test)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label, probabilities

def get_accuracy(trained_model, X_test, y_test, encode=False):
    # list to hold the actual results of X_test and a list to hold predicted results
    actual_res = []
    predicted_res = [item for sublist in y_test.values for item in sublist]

    # Check if you need to encode X_test
    if encode is True:
        X_test, foo = encode_string_columns(X_test, X_test.iloc[0,:-1])

    for row in range(len(X_test)):
        pred_row = give_predict(trained_model, X_test.iloc[row, :-1])
        actual_res.append(pred_row[0])

    correct = 0
    for i in range(len(actual_res)):
        if actual_res[i] == predicted_res[i]:
            correct += 1
    return correct / float(len(actual_res)) * 100.0

def main(): # TODO: figure out how this will work terminal wise
    # Retrieve Data
    heart_data = pd.read_csv('data/heart.csv')
    # We want to create both training and testing datasets, we will use test_train_split to do this
    X_train, X_test, y_train, y_test = train_test_split(heart_data,
        heart_data.iloc[:, -1:], test_size=0.30, random_state=42)
    print(len(X_test))
    model = summarize_by_class(X_train)
    pred = give_predict(model, X_test.iloc[3,:-1])
    print('Heart Classifier')
    print('Data=%s, Predicted: %s' % (X_test.iloc[3, :-1].values.tolist(), pred))
    print('Acutal = %s' % (y_test.iloc[3,0]))
    # print(X_test)
    print(get_accuracy(model, X_test, y_test))

    data = pd.read_csv("data/balloons.csv")
    X_train1, X_test1, y_train1, y_test1 = train_test_split(data,
            data.iloc[:, -1:], test_size=0.30, random_state=42)
    model1, test_new = summarize_by_class(X_train1, X_test1.iloc[0, :-1])
    label1 = give_predict(model1, test_new)
    print('Ballon Classifier')
    print('Data=%s, Predicted: %s' % (X_test1.iloc[0, :-1].values.tolist(), label1))
    print(get_accuracy(model1, X_test1, y_test1, True))


    # # Test Data for prediction
    # test = [60, 1, 0, 130, 283, 0, 0, 108, 1, 1.5, 1, 3, 2]
    # model = summarize_by_class(heart_data)
    # # predict the label
    # label = give_predict(model, test)
    # print('Heart Classifier')
    # print('Data=%s, Predicted: %s' % (test, label))
    #
    # # Retrieve data
    # data1 = pd.read_csv("data/balloons.csv")
    # '''
    # In cases where there is a test that contains categorical
    # data we need to give summarize_by_class a second argument.
    # The argument will be the test data the you like to use agianst
    # the Bayes prediction.
    # '''
    # test1 = ['YeLlOw', 'SMALL', 'DIP', 'CHILD']
    # model1, test_new = summarize_by_class(data1, test1)
    # label1 = give_predict(model1, test_new)
    # print('Ballon Classifier')
    # print('Data=%s, Predicted: %s' % (test1[:-1], label1))
if __name__ == "__main__":
    main()
