# Custom functions for heart disease detection project

# Imports
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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


def get_safest_thresh(model, X_train, y_train):
    """
    Description: generates data frame with false and true positive rates for varying thresholds and a score of the highest threshold that leaves no false negatives
    Inputs: model = model used for predictions
            X = data frame of predictor features
            y = data frame of target feature values
    Outputs: thresh_df = data frame of true and false positive rates for various threshold values
             safest_thresh = highest threshold that still yields no false negatives
    """
    probs = model.predict_proba(X_train)[:,1]
    fpr, tpr, thresh = roc_curve(y_train, probs)
    thresh_df = pd.DataFrame({'Threshold':thresh,
                             'False Positive Rate':fpr,
                             'True Positive Rate':tpr})
    rf_safe_thresh = thresh_df.loc[thresh_df['True Positive Rate']==1]
    safest_thesh = rf_safe_thresh['Threshold'].max()
    return safest_thesh


def get_scores(model, X, y, thresh=0.5, format='df', name='Model Scores', safe=False, X_train=None, y_train=None):
    """
    Description: uses a model to predict ys from Xs and then scores those preductions and returns the scores
    Inputs: model = the model to predict with
            X = predictor feature data, could be train, validation, or test
            y = target feature data, could be train, validation, or test
            thresh = prediction probabilities above this threshold will be considered positive. 0.5 will return the model's best guess
            format = either 'df' to return data frame or 'dict' to return dictionary
            name = name of model, can be helpful if you want to combine dataframes of multiple models' scores
            safe = boolean, if True uses threshold with maximum recall
            X_train = if safe is True you must input training data
            y_train = if safe is True you must input training data
    Outputs: df = data frame containing scores
    """
    if safe==True:
        thresh = get_safest_thresh(model, X_train, y_train)
    y_probs = model.predict_proba(X)[:,1]
    y_pred = (y_probs>=thresh).astype(bool)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_probs)
    score_dict = {'Accuracy':acc,
                  'Precision':prec,
                  'Recall':recall,
                  'F1':f1,
                  'AUC':auc}
    if format == 'dict':
        return score_dict
    else:
        df = pd.DataFrame(score_dict, index=[name])
        return df

    
def score_models(models_dict, X, y, safe=False, X_train=None, y_train=None):
    """
    Description: generates a data frame of scores for several models
    Inputs: models_dict =  dictionary of model_name-model pairs
            X = predictor features for models to predict with
            y = target feature for models to be evaluated with
            safe = boolean, if True uses threshold with max recall
            X_train = if safe is True you must input training data
            y_train = if safe is True you must input training data
    Outputs: scores_df = data frame of all model scores
    """
    first_model = list(models_dict.keys())[0]
    scores_df = get_scores(models_dict[first_model], X, y, name=first_model, safe=safe, X_train=X_train, y_train=y_train)
    for model in list(models_dict.keys())[1:]:
        model_scores = get_scores(models_dict[model], X, y, name=model, safe=safe, X_train=X_train, y_train=y_train)
        scores_df = pd.concat([scores_df, model_scores])
    return scores_df


def plot_confusion(model, X, y, thresh=0.5, name='', safe=False, X_train=None, y_train=None):
    """
    Description: plots confusion matrix
    Inputs: model = model used to make predictions
            X = predictor features used to make predictions
            y = target feature used to evaluate predictions
            thresh = prediction probabilities above this threshold will be considered positive. 0.5 will return the model's best guess
            name = name of model to be used in plot title
            safe = boolean, if True uses threshold with maximum recall
            X_train = if safe is True you must input training data
            y_train = if safe is True you must input training data
    Outputs: none, just plots matrix
    """
    if safe==True:
        thresh = get_safest_thresh(model, X_train, y_train)
    probs = model.predict_proba(X)[:,1]
    pred = (probs >= thresh).astype(bool)
    conf_arry = confusion_matrix(y, pred, labels=None, sample_weight=None, normalize=None)
    cm = ConfusionMatrixDisplay(confusion_matrix = conf_arry, display_labels=['Healthy', 'Diseased'])
    title = name + ' Confusion Matrix'
    cm.plot()
    plt.title(title, {'size':17})
    plt.show()


def plot_roc(models_dict, X, y):
    """
    Description: uses a dictionary of name-model pairs and predictor and target data to plot a roc curve
    Inputs: models_dict = dictionary with name-model pairs
            X = predcitor feature data for models to predict with
            y = target feature data to evaluate predictions with
    Outputs: none, just plots curve
    """
    for model in list(models_dict.keys()):
        probs = models_dict[model].predict_proba(X)[:,1]
        fpr, tpr, thresh = roc_curve(y, probs)
        plt.plot(fpr, tpr, marker = '.', label = model)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def plot_pr(models_dict, X, y):
    """
    Description: uses a dictionary of name-model pairs and predictor and target data to plot a precision-recall curve
    Inputs: models_dict = dictionary with name-model pairs
            X = predcitor feature data for models to predict with
            y = target feature data to evaluate predictions with
    Outputs: none, just plots curve
    """
    for model in list(models_dict.keys()):
        probs = models_dict[model].predict_proba(X)[:,1]
        precision, recall, _ = precision_recall_curve(y, probs)
        plt.plot(recall, precision, marker='.', label = model)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()