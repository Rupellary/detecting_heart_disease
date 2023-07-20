# Custom functions for heart disease detection project

# Imports
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score


def drop_data(df, col, value):
    """
    Description: deletes observations with a given value in a given column of a given dataframe
    Inputs: df = data frame from which to remove observations
            col = the column to be searched for the value
            value =  the value that must be present in order for the column to be deleted
    Outputs: new_df = data frame with problem observations removed
    """
    new_df = df.loc[df[col] != value]
    return new_df


def string_to_Na(df, string):
    """
    Description: converts cells with a given string value into None in a given data frame
    Inputs: df =  data frame to change
            string = string a cell must have to be converted into None
    Outputs: df = updated data frame
    """
    for col in df.select_dtypes(exclude='number').columns:
        df.loc[df[col]==string, col]=None
    return df

def list_indexes(search_list, indexed_list):
    """
    Description: uses a list of strings and finds the indexes of those strings in another list
    Inputs: search_list = list of strings to find the indexes of
            indexed_list = list with desired indexes
    Outputs: indexes = list of indexes
    """
    indexes = [index for index, item in enumerate(indexed_list) if item in search_list]
    return indexes


def get_scores(model, X, y, format='df', name='Model Scores'):
    """
    Description: uses a model to predict ys from Xs and then scores those preductions and returns the scores
    Inputs: model = the model to predict with
            X = predictor feature data, could be train, validation, or test
            y = target feature data, could be train, validation, or test
            format = either 'df' to return data frame or 'dict' to return dictionary
            name = name of model, can be helpful if you want to combine dataframes of multiple models' scores
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    score_dict = {'Accuracy':acc,
                  'Precision':prec,
                  'Recall':recall,
                  'F1':f1}
    if format == 'dict':
        return score_dict
    else:
        df = pd.DataFrame(score_dict, index=[name])
        return df
    
def plot_confusion(model, X, y):
    """
    Description:
    """
    pred = model.predict(X)
    conf_arry = confusion_matrix(y, pred, labels=None, sample_weight=None, normalize=None)
    ConfusionMatrixDisplay(confusion_matrix = conf_arry, display_labels=['Healthy', 'Diseased']).plot()