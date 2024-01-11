# Import Common Libraies
import pandas as pd 
import numpy as np
import os

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score,KFold, StratifiedKFold, ShuffleSplit
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif


# Own Library
from Own_Library.Classification_Reports import classification_reports
from Own_Library.GroupBarPlt_test import groupBarPlot

scaler = RobustScaler()

def find_reports(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cldf, _ = classification_reports(y_test, y_pred)
    return cldf


def best_find(matrix):
    # print("Reports matrix (Row wise)\n", matrix)
    index = np.zeros(len(matrix), dtype=int)
    for i in range(len(matrix[0])):
        # index[np.argmax(matrix[:, i])] += 1
        index[np.argwhere(matrix[:, i] == np.amax(matrix[:, i])).flatten()] += 1
    print("Technique matrix with total best reports:", index)
    return np.argmax(index)


# 1. All SPLIT CHECKING
def Splitings(model, X, y, splits, metrics, algoname, dim =False):
    m = [0 for _ in range(len(splits))]
    dict1= {'Metrics': metrics}
    for i in range(len(splits)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= splits[i], random_state=42)
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        if dim:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        cldf = find_reports(model, X_train, X_test, y_train, y_test)
        # m[i] = cldf[metrics].iloc[-1].tolist()
        m[i]= cldf[metrics].values[-1]
        
        dict1[f'Train ratio: {(1-splits[i])}, Test ratio: {splits[i]}'] = m[i]
    
    plotdf = pd.DataFrame(dict1)
    groupBarPlot(plotname= 'Train-Test Split Ratio', algoname= algoname, df=plotdf)

    best_split = splits[best_find(np.array(m))]
    # print("Spliting Classification Metrics DataFrame: \n", plotdf)
    print("Best test size: ", best_split)

    return best_split, plotdf


# 2. ALL FEATURE SELECTION CHECKING
def Feature_Selections(model, data, best_split, metrics, algoname, dim = False):
    feature_selections = ['Without', 'Pearson Correlation', 'Mutual Info Classif', 'Chi_sqaure']
    y = data.iloc[:,-1]

    X_vector= []
    m = [0 for _ in range(len(feature_selections))]
    dict1= {'Metrics': metrics}

    for i in range(len(feature_selections)):
        # Without Feture Selection
        X = data.iloc[:,:-1] # DataFrame

        # Pearson Correlation
        if i == 1:          
            corr_matt = data.corr(method='pearson')[[data.columns[-1]]].values 
            sel_cols1 = np.where(abs(corr_matt)>0.05)[0]
            del_cols1 = np.where(abs(corr_matt)<0.05)[0]
            X = X[data.columns[sel_cols1[:-1]]]

        # Mutual_info_classif 
        if i == 2:
            fvalue_selector = SelectKBest(mutual_info_classif, k= len(X_vector[1][0])) # k = 5 or len(X_vector[1][0] 
            fvalue_selector.fit(X, y)
            sel_cols2 = X.columns[fvalue_selector.get_support()]
            del_cols2 =[i for i in X.columns if i not in sel_cols2]
            X = X[sel_cols2]        

        # Chi_square
        if i == 3:
            fvalue_selector = SelectKBest(chi2, k= len(X_vector[1][0])) # k = 5 or len(X_vector[1][0] 
            fvalue_selector.fit(X, y)
            sel_cols3 = X.columns[fvalue_selector.get_support()]
            del_cols3 =[i for i in X.columns if i not in sel_cols3]
            X = X[sel_cols3]

        X = scaler.fit_transform(X)
        X_vector.append(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= best_split, random_state=42)
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        if dim:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        cldf = find_reports(model, X_train, X_test, y_train, y_test)
        m[i]= cldf[metrics].values[-1] 
        dict1[feature_selections[i]] = m[i]

    plotdf = pd.DataFrame(dict1)
    groupBarPlot(plotname= 'Feature Selection Techniques', algoname= algoname, df=plotdf)

    best_feature_selection = best_find(np.array(m))
    # print("Feature Selections Classification Metrics DataFrame: \n", plotdf)
    print('Best Feature Selection Technique:', feature_selections[best_feature_selection])

    if best_feature_selection == 0:
        return X_vector[0], plotdf, 'Not Applicable'
    elif best_feature_selection == 1:
        return X_vector[1], plotdf, data.columns[del_cols1]
    elif best_feature_selection == 2: 
        return X_vector[2], plotdf, del_cols2
    else: return X_vector[3], plotdf, del_cols3


# 3. All CV CHECKING
def Cross_Validations(model, featX, y, best_split, metrics, algoname, dim = False):
    n_splits = int(100/(best_split*100))
    cvs = [ KFold(n_splits, shuffle = True, random_state=42), 
            StratifiedKFold(n_splits, shuffle = True, random_state=42),
            ShuffleSplit(n_splits, test_size= best_split, random_state=42)] # n_splits, test_size= 0.1, is bydefault in shuffleSplit
    cvn=['KFold','StratifiedKFold','ShuffleSplit']

    if dim:
        featX = np.reshape(featX, (featX.shape[0], featX.shape[1], 1))
        
    dict1= {'Metrics': metrics}
    m = [[0 for _ in range(len(metrics))] for _ in range(len(cvs))]
    for i in range(len(cvs)):
        for j in range(len(metrics)):

            scores = cross_val_score(model, featX, y, cv = cvs[i], scoring=metrics[j])
            m[i][j] = scores.mean()
            
        dict1[cvn[i]] = m[i]
    
    plotdf = pd.DataFrame(dict1)
    groupBarPlot(plotname='Cross-Validation Techiniques', algoname=algoname, df=plotdf)
    
    best_cv = cvs[best_find(np.array(m))]
    # print("Cross Validations Classification Metrics DataFrame: \n", plotdf)
    print(f"Best CV technique: {best_cv}")

    return best_cv, plotdf