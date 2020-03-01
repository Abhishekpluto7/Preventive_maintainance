import dataprep
import pandas as pd
import fileOperations
import math
from sklearn.ensemble import RandomForestRegressor

def trainAndEvaluateModel(parametets):
    X_train, x_test, Y_train, y_test = dataprep.getDataForModelTraining(parametets.trainFilePath, parametets.trainOutputPath)


    random_forest_regressor = RandomForestRegressor(n_estimators=15)
    print('\n\n\n...........TRAINING RANDOM FOREST REGRESSOR..........\n\n\n')
    random_forest_regressor.fit(X_train, Y_train)
    #random_forest_regressor = fileOperations.local_read_trained_model('Dataset/model.pkl')
    #fileOperations.gcs_write_trained_model(parametets.outputFilePath, random_forest_regressor)
    print('\n\n\nRANDOM FOREST REGRESSION TRAINING SUCESSFULL\n\n\n')
    train_subset_prediction = random_forest_regressor.predict(x_test)

    #print the metrics for train subset
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    print('\n\n\nRegression Metrics for train subset prediction')
    print('Mean Squared Error:   ',mean_squared_error(train_subset_prediction, y_test))
    print('Root Mean Squared Error:   ',math.sqrt(mean_squared_error(train_subset_prediction, y_test)))
    print('Mean Absolute Error', mean_absolute_error(train_subset_prediction, y_test))
    print('R2 Score', r2_score(train_subset_prediction,y_test))

    #print('Accuracy ', random_forest_regressor.score(train_subset_prediction, y_test))


    #evaluate on test dataset
    def getEvaluationTruthValues(evaluation_rul_prediction,expected_rul_dataframe):
        predicted_dataframe = X
        predicted_dataframe['RUL'] = evaluation_rul_prediction
        #extract the predicted RUL for the each engine from the predicted dataframe
        rul = pd.DataFrame(predicted_dataframe.groupby('Id')['Cycle'].max()).reset_index()
        rul.columns=['Id','Cycle']
        remainingcycle = list()
        i=1
        for cycle in (rul['Cycle'].values):
            remainingcycle.append(predicted_dataframe.loc[(predicted_dataframe['Id'] == i) & (predicted_dataframe['Cycle']==cycle)]['RUL'].values[0])
            i = i+1

        #extract the expected RUL
        expected_rul = pd.DataFrame(expected_rul_dataframe.groupby('Id')['Cycle'].max()).reset_index()
        #expected_rul_dataframe.columns=['Id','Cycle']
        expected_remainingcycle = list()
        j=1
        for cycle in (expected_rul['Cycle'].values):
            expected_remainingcycle.append(expected_rul_dataframe.loc[(expected_rul_dataframe['Id'] == j) & (expected_rul_dataframe['Cycle']==cycle)]['RUL'].values[0])
            j= j+1

        print("\n\n\nPredicted RUL")    
        print(remainingcycle)
        print("\n\nexpected RUL")
        print(expected_remainingcycle)
        print('\n\n\nRegression Metrics for train subset prediction')
        mse = mean_squared_error(remainingcycle, expected_remainingcycle)
        rmse = math.sqrt(mean_squared_error(remainingcycle, expected_remainingcycle))
        mae = mean_absolute_error(remainingcycle, expected_remainingcycle)
        r2 = r2_score(remainingcycle, expected_remainingcycle)
        metrics_string = 'Mean Squared Error:\t\t\t'+str(mse)+'\n\nRoot Mean Squared Error:\t\t'+str(rmse)+'\n\nMean Absolute Error:\t\t\t'+str(mae)+'\n\nR2 Score:\t\t\t\t'+str(r2)

        print('Mean Squared Error:   ',mse)
        print('Root Mean Squared Error:   ',rmse)
        print('Mean Absolute Error', mae)
        print('R2 Score', r2)
        #print('Accuracy ', random_forest_regressor.summary())
        print(metrics_string)
        result_dataframe = pd.DataFrame(remainingcycle,columns=['Predicted RUL'])
        fileOperations.gcs_write_csv_file(parametets.outputFilePath+'/randomforestregression_output.csv', result_dataframe)
        fileOperations.gcs_write_text_file(parametets.outputFilePath+'/randomfoestregression_metrics.txt', metrics_string)




        return remainingcycle


    print('EVALUATIONG THE TEST SET......')
    X, Y = dataprep.getDataForEvaluation(parametets.testFilePath,parametets.testOutputPath)
    evaluation_rul_prediction = random_forest_regressor.predict(X)
    remaining_useful_lifcycles = getEvaluationTruthValues(evaluation_rul_prediction,Y)