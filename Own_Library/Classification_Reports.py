from math import sqrt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def classification_reports(actual, predicted, category = None):
    
    # Create the confusion matrix
    classes = np.unique(actual) # extract the different classes
    cm = np.zeros((len(classes), len(classes))) # initialize the confusion matrix with zeros
    for i in range(len(classes)):
        for j in range(len(classes)):
            cm[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))


    # Define Category Name
    if category == None:
        category = [str(i) for i in range(len(cm))] 


    # Define the Classification metrics Dataframe    
    classif_df = pd.DataFrame({'Category':pd.Series(dtype='category'),
                             'Precision':pd.Series(dtype='float32'),
                             'Recall':pd.Series(dtype='float32'),
                             'F1_Score':pd.Series(dtype='float32'),
                             'F1_Measure': pd.Series(dtype='float32'),
                             'Specificity': pd.Series(dtype='float32'),
                             'Negative_Predictive_Value': pd.Series(dtype='float32'),
                             'False_Positive_Rate': pd.Series(dtype='float32'),
                             'False_Negative_Rate': pd.Series(dtype='float32'),
                             'False_Discovery_Rate': pd.Series(dtype='float32'),
                             'Critical_Success_Index': pd.Series(dtype='float32'),
                             'Fowlkes_Mallows_Index': pd.Series(dtype='float32'),

                             'False_Omission_Rate': pd.Series(dtype='float32'),
                             'Positive_Likelihood_Ratio': pd.Series(dtype='float32'),
                             'Negative_Likelihood_Ratio': pd.Series(dtype='float32'),
                             'Prevalence_Threshold': pd.Series(dtype='float32'),
                             'Diagnostic_Odds_Ratio': pd.Series(dtype='float64'),
                            
                             'Balanced_Accuracy': pd.Series(dtype='float32'),
                             'Mathews_Correlation_Coefficient': pd.Series(dtype='float64'),
                             'Bookmaker_Informedness': pd.Series(dtype='float32'),
                             'Markedness': pd.Series(dtype='float32'),
                             'Accuracy': pd.Series(dtype='float32'),
                             'Support': pd.Series(dtype='int')})
    

    # Define the Confusion matrix element Dataframe
    perconf_df = pd.DataFrame({'Category': pd.Series(dtype='category'), 
                            'True Positive': pd.Series(dtype='int'),
                            'False Positive':pd.Series(dtype='int'),
                            'False Negative':pd.Series(dtype='int'),
                            'True Negative':pd.Series(dtype='int'),})
    

    # Fillup Dataframes
    totalSamp = np.sum(cm)
    for cls in range(len(cm)):
        TP = cm[cls][cls]                        # True positive
        FN = sum(cm[cls]) - TP                   # False Negative
        FP = sum(cm[i][cls] \
             for i in range(len(cm))) - TP       # False Positive
        TN = totalSamp-(TP+FP+FN)                # True Negative
        perconf_df.loc[len(perconf_df.index)] = [category[cls], TP, FP, FN, TN]
        
        PPV = (TP)/(TP+FP)                       # Precision or Positive_Precdictive_Value (PPV)
        TPR = TP/(TP+FN)                         # Recall or Sensitivity or True_Positive_Rate (TPR) or Hit_Rate
        F1_S = (2*PPV*TPR)/(PPV+TPR)             # F1 Score or Harmonic Mean
        F1_M = (PPV+TPR)/2                       # F1 Measure
        TNR = TN/(TN+FP)                         # Specificity or True_Negative_Rate(TNR) or Selectivity
        NPV = TN/(TN+FN)                         # Negative_Predictive_Value
        FPR = FP/(FP+TN)                         # False_Positive_Rate
        FNR = FN/(TP+FN)                         # False_Negative_Rate or Miss_Rate
        FDR = FP/(TP+FP)                         # False_Discovery_Rate
        CSI = TP/(TP+FN+FP)                      # Critical_Success_Index or Threat_Score(TS)
        FM = sqrt(PPV*TPR)                       # Fowlkes_Mallows_Index

        FOR = FN/(FN+TN)                         # False_Omission_Rate
        PLR = TPR/FPR                            # Positive_Likelihood_Ratio
        NLR = FNR/TNR                            # Negative_Likelihood_Ratio
        PT = sqrt(FPR)/(sqrt(TPR)+sqrt(FPR))     # Prevalence_Threshold
        DOR = PLR/NLR                            # Diagnostic_Odds_Ratio
        
        BA = (TPR+TNR)/2                         # Balanced_Accuracy
        MCC = (TP*TN-FP*FN)/(sqrt(\
              (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))  # Mathews_Correlation_Coefficient
        BI = TPR+TNR-1                           # Bookmaker_Informedness or Informedness
        MK = PPV+NPV-1                           # Markedness or deltaP
        Support = (TP+FN)                        # Support
        classif_df.loc[len(classif_df.index)] = [category[cls], PPV, TPR, F1_S, F1_M, 
                                                 TNR, NPV, FPR, FNR, FDR, CSI, FM,
                                                 FOR, PLR, NLR, PT, DOR, 
                                                 BA, MCC, BI, MK, '', Support] #


    # Define the Accuracy
    tr = sum(cm[i][i] for i in range(len(cm)))
    Acc = tr/totalSamp                           # Accuracy

    # Fillup Macro Average Row
    classif_df.loc[len(classif_df.index)] = ['Macro_Avg'] + \
                                            classif_df.iloc[:, 1:-2].mean().to_list() + \
                                            [Acc] + \
                                            [totalSamp]

    # Print or return Dataframes
    # print(classif_df.to_string(index=False)) 
    return classif_df, perconf_df
       


# For verifying purpose
if __name__ == "__main__":
    # Import libraries
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    # Load dataset into the memory
    data = pd.read_csv('Extra\EEG_Eye_State.csv')

    # Preprocess dataset
    data.dropna(inplace=True)
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model fit and predict
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print reports
    classif_df, perconf_df = classification_reports(y_test, y_pred)
    print(classif_df)
    print()
    print(perconf_df)
    print("\nSklearn classif repo:\n", classification_report(y_test, y_pred))