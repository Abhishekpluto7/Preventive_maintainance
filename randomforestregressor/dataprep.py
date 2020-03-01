import fileOperations
import pandas as pd

#feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def feature_selection(fs_dataframe):
    print('\n\n..........PERFORMING FEATURE SELECTION USING CHI2.....')
    dataset = fs_dataframe.copy()
    X = dataset.drop(['RUL'], axis=1)
    Y= dataset['RUL']
    bestfeatures = SelectKBest(score_func=chi2, k=18)
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    featureScores = featureScores.sort_values(['Score'])
    features_to_drop = featureScores[featureScores['Score'].isnull()]
    print('\n\nFeatures that can be removed are \n',features_to_drop['Specs'].values)
    return list(features_to_drop['Specs'])
    

def remove_features(rf_dataframe, features_to_drop):
    print('\n\n{}\n Removing above Features............:\n '.format(features_to_drop))
    rf_dataframe = rf_dataframe.drop(features_to_drop, axis=1)
    return rf_dataframe



#feature scaling
from sklearn.preprocessing import MinMaxScaler
def feature_scaling(dataframe, columns_to_normalize):
    print('\n\n....Performing features scaling.....')
    x = dataframe[columns_to_normalize]
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns= columns_to_normalize)
    df['Id'] = dataframe['Id']
    df['Cycle'] = dataframe['Cycle']
    df['RUL'] = dataframe['RUL']
    #df['label1'] = dataframe['label1']
    #df['label2'] =  dataframe['label2']
    return df



def getDataForModelTraining(traininputpath, trainoutputpath):
    #read csv file
    print('\n\n .....Collecting data for Model training.....')
    print(traininputpath)
    dataframe = fileOperations.gcs_read_csv_file(traininputpath)
    dataframe['RUL'] = fileOperations.gcs_read_csv_file(trainoutputpath)
    #feature scaling
    columns_to_scale = [x for x in dataframe.columns if(not(x in ['Id','Cycle','RUL','label1','label2']))]
    scaled_dataframe = feature_scaling(dataframe, columns_to_scale)
    #feature selection
    features_to_remove = feature_selection(scaled_dataframe)
    clean_dataframe = remove_features(scaled_dataframe, features_to_remove)
    
    #split the dataset and send back
    from sklearn.model_selection import train_test_split
    dataset = clean_dataframe.copy()
    X = dataset.drop(['RUL'], axis=1)
    Y = pd.DataFrame()
    Y['Id'] = dataset['Id']
    Y['RUL'] = dataset['RUL']
    k = Y.groupby('Id').head(len(Y['Id']))
    #print(Y['RUL'])

    X_train, x_test, Y_train, y_test = train_test_split(X.groupby('Id').head(len(X['Id'])),Y['RUL'], test_size=0.20)

    return X_train,x_test,Y_train,y_test


def getDataForEvaluation(testinputpath, testoutputpath):
    print('\n\n.....Collecting data for model Evaluation.....')
    #read csv file
    dataframe = fileOperations.gcs_read_csv_file(testinputpath)
    dataframe['RUL'] = fileOperations.gcs_read_csv_file(testoutputpath)
    #feature scaling
    columns_to_scale = [x for x in dataframe.columns if(not(x in ['Id','Cycle','RUL','label1','label2']))]
    scaled_dataframe = feature_scaling(dataframe, columns_to_scale)
    #feature selection
    features_to_remove = feature_selection(scaled_dataframe)
    clean_dataframe = remove_features(scaled_dataframe, features_to_remove)
    dataset =  clean_dataframe.copy()
    X = dataset.drop(['RUL'], axis=1)
    Y = pd.DataFrame()
    Y['Id'] = dataset['Id']
    Y['Cycle'] = dataset['Cycle']
    Y['RUL'] = dataset['RUL']
    #print(Y['RUL'])
    return X,Y






