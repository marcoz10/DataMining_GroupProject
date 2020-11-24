# Imports for this file
from random import randrange
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
    # for x in user_test:
    #     x.append(None)
    print(user_test)
    column_names = dataframe.columns.values.tolist()
    frame = pd.DataFrame(user_test, columns=column_names)
    # print(frame)
    dataframe = dataframe.append(frame, ignore_index= True)

    # Use the list and loop through the columns and encode them
    for col in list_of_string_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
        dataframe[col] = dataframe[col].astype(int)
    new_user_test = []
    new_user_test.append(dataframe.iloc[-1,:-1].to_list())
    dataframe = dataframe.iloc[:-1, :]
    return dataframe, new_user_test

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
def summarize_by_class(dataset, user_test = None):
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
        print(summaries)
        return summaries , new_user_test
    else:
        return summaries

'''
calculate_probaility(x, mean, stdev): this function will calculate the
bayes probability of an event occuring given x, x's mean, and x's stdev 
'''
def calculate_probability(x, mean, stdev):
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
    return best_label

# Naive Bayes Algorithm
def naive_bayes(train, test, encode = False):
    if encode == True:
        summarize, test = summarize_by_class(train, test)
        print(test)
    else:
        summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = give_predict(summarize, row)
        predictions.append(output)
    print('Data = ' + str(test) + 'Predicted = ' + str(predictions))
    return(predictions)

def cross_validate_bayes(dataset, n_folds):
    data_split = []
    data_copy = dataset.values.tolist()
    fold_size = int(len(dataset)/n_folds)
    for f in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            ind = randrange(len(data_copy))
            fold.append(data_copy.pop(ind))
        data_split.append(fold)
    return data_split

def calculate_accuracy_bayes(actual, pred):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            correct += 1
    return correct / float(len(actual)) * 100

def evaluate_bayes_algorithm(dataset, algorithm, n_folds, encode=False):
    folds = cross_validate_bayes(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        column_names = dataset.columns.values.tolist()
        frame = pd.DataFrame(train_set, columns=column_names)
        if encode == True:
            predicted = algorithm(frame,test_set,True)
        else:
            predicted = algorithm(frame, test_set)
        actual = [row[-1] for row in fold]
        accuracy = calculate_accuracy_bayes(actual,predicted)
        scores.append(accuracy)
    return scores


def main(): # TODO: figure out how this will work terminal wise
    # Retrieve Data
    heart_data = pd.read_csv('data/heart.csv')
    # Test Data for prediction
    test = [[60, 1, 0, 130, 283, 0, 0, 108, 1, 1.5, 1, 3, 2]]
    X_train, X_test, y_train, y_test = train_test_split(heart_data, heart_data.iloc[:, -1:], test_size=0.30, random_state=42)
    n_folds = 10

    scores = evaluate_bayes_algorithm(X_train,naive_bayes, n_folds)
    print(' Scores: %s ' % scores)
    print('Accuracy: %.3f%% ' % (sum(scores) / float(len(scores))))
    naive_bayes(X_train, test)

    # # Retrieve data
    data1 = pd.read_csv("data/balloons.csv")
    # test1 = [['YeLlOw', 'SMALL', 'DIP', 'CHILD']]
    X_train, X_test, y_train, y_test = train_test_split(data1, data1.iloc[:, -1:], test_size=0.30,random_state=42)
    n_folds = 10

    scores = evaluate_bayes_algorithm(X_train,naive_bayes, n_folds, True)
    print(' Scores: %s ' % scores)
    print(' Mean Accuracy: %.3f%% ' % (sum(scores) / float(len(scores))))
    # naive_bayes(X_train, test1, True)
if __name__ == "__main__":
    main()
