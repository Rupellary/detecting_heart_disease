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
    # create and return a data frame without observations with the input value in the input column
    new_df = df.loc[df[col] != value]
    return new_df


def string_to_Na(df, string):
    """
    Description: converts cells with a given string value into None in a given data frame
    Inputs: df =  data frame to change
            string = string a cell must have to be converted into None
    Outputs: df = updated data frame
    """
    # loop through columns that can contain strings
    for col in df.select_dtypes(exclude='number').columns:
        #select observations with the input string and set them to None
        df.loc[df[col]==string, col]=None
    return df


def list_indexes(search_list, indexed_list):
    """
    Description: uses a list of strings and finds the indexes of those strings in another list
    Inputs: search_list = list of strings to find the indexes of
            indexed_list = list with desired indexes
    Outputs: indexes = list of indexes
    """
    # get index-item pairs from indexed_list with enumerate, 
    # loop through them and check if the item is in the search list, 
    # if it is then include the index in the output list
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
    # make probabilistic predictions with model
    probs = model.predict_proba(X_train)[:,1] #select only probabilities
    # calculate roc curve with true positive and false positive rates at various thresholds
    fpr, tpr, thresh = roc_curve(y_train, probs)
    # turn roc array into easily searchable data frame
    thresh_df = pd.DataFrame({'Threshold':thresh,
                             'False Positive Rate':fpr,
                             'True Positive Rate':tpr})
    # create data frame with a subset of the curve where there are no false negatives
    rf_safe_thresh = thresh_df.loc[thresh_df['True Positive Rate']==1]
    # select the highest threshold within the subset of those with max recall
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
        # when safe is set to True, get the threshold that has maximum recall with the training data
        thresh = get_safest_thresh(model, X_train, y_train)
    # make probablistic predictions with model
    y_probs = model.predict_proba(X)[:,1] #select only probabilities
    # convert probablities into boolean predictions according to threshold
    y_pred = (y_probs>=thresh).astype(bool)
    # calculate scores and store them in a dictionary
    score_dict = {'Accuracy':accuracy_score(y, y_pred),
                  'Precision':precision_score(y, y_pred),
                  'Recall':recall_score(y, y_pred),
                  'F1':f1_score(y, y_pred),
                  'AUC':roc_auc_score(y, y_probs)}
    # return in desired format
    if format == 'dict':
        return score_dict
    else:
        df = pd.DataFrame(score_dict, index=[name])
        return df

    
def score_models(models_dict, X, y, thresh = 0.5, safe=False, X_train=None, y_train=None):
    """
    Description: generates a data frame of scores for several models
    Inputs: models_dict =  dictionary of model_name-model pairs
            X = predictor features for models to predict with
            y = target feature for models to be evaluated with
            thresh = prediction probabilities above this threshold will be considered positive. 0.5 will return the model's best guess
            safe = boolean, if True uses threshold with max recall
            X_train = if safe is True you must input training data
            y_train = if safe is True you must input training data
    Outputs: scores_df = data frame of all model scores
    """
    # get model names and add details if there is a threshold or max recall
    models = list(models_dict.keys())
    model_names = models
    if safe == True:
        model_names = [name + ' (max recall)' for name in model_names]
    elif thresh != 0.5:
        model_names = [name + f' (with threshold of {thresh})' for name in model_names]
    # initialize data frame with correct columns by just scoring first model
    scores_df = get_scores(models_dict[models[0]], X, y, name=model_names[0], thresh=thresh, safe=safe, X_train=X_train, y_train=y_train)
    # loop through remaining models, concatonating their scores with the results of the first
    for i in range(1,len(models)):
        model_scores = get_scores(models_dict[models[i]], X, y, name=model_names[i], thresh=thresh, safe=safe, X_train=X_train, y_train=y_train)
        scores_df = pd.concat([scores_df, model_scores])
    return scores_df

def best_vs_safe(models_dict, X, y, X_train, y_train, thresh=0.5):
    """
    Description: scores models with the best prediction, the safest prediction, and optionally predictions at an input threshold and returns the scores in a data frame
    Inputs: models_dict =  dictionary of model_name-model pairs
            X = predictor features for models to predict with
            y = target feature for models to be evaluated with
            X_train = training data predictor features
            y_train = training data target feature
            thresh = prediction probabilities above this threshold will be considered positive. 0.5 will return the model's best guess
    Outputs: compare_df = data frame with all scores
    """
    best_df = score_models(models_dict, X, y, thresh = 0.5, safe=False, X_train=None, y_train=None)
    safe_df = score_models(models_dict, X, y, safe=True, X_train=X_train, y_train=y_train)
    compare_df = pd.concat([best_df, safe_df])
    if thresh != 0.5:
        thresh_df = score_models(models_dict, X, y, thresh=thresh, safe=False, X_train=X_train, y_train=y_train)
        compare_df = pd.concat([compare_df, thresh_df])
    compare_df.sort_index(axis=0, inplace=True)
    return compare_df
    
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
        # when safe is set to True, get the threshold that has maximum recall with the training data
        thresh = get_safest_thresh(model, X_train, y_train)
    # make probablistic predictions with model
    probs = model.predict_proba(X)[:,1] #select only probabilities
    # convert probabilities to boolean predictions according to threshold
    pred = (probs >= thresh).astype(bool)
    # use sklearn functions to create and display confusion matrix
    conf_arry = confusion_matrix(y, pred, labels=None, sample_weight=None, normalize=None)
    ConfusionMatrixDisplay(confusion_matrix = conf_arry, display_labels=['Healthy', 'Diseased']).plot()
    # add model name to plot title
    title = name + ' Confusion Matrix'
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
    # loop through models in dictionary, calculate their ROC curves using sklearn function and then plot
    for model in list(models_dict.keys()):
        probs = models_dict[model].predict_proba(X)[:,1] #select only probabilities
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
    # loop through models in dictionary, calculate their precision-recall curves using sklearn function and then plot
    for model in list(models_dict.keys()):
        probs = models_dict[model].predict_proba(X)[:,1] #select only probabilities
        precision, recall, _ = precision_recall_curve(y, probs)
        plt.plot(recall, precision, marker='.', label = model)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()