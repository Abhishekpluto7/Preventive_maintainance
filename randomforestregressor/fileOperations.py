import pandas as pd 
import pickle
import tensorflow as tf

#Read and write csv operations using tensorflow to and form google cloud storage bucket
def gcs_read_csv_file(filepath, opts=None):
    with tf.io.gfile.GFile(filepath) as f:
        if(opts):
            df = pd.read_csv(f,opts)
        else:
           df = pd.read_csv(f)
    print('CSV file read sucessfull')
    return df

def gcs_write_csv_file(filepath, dataframe, opts=None):
    with tf.io.gfile.GFile(filepath, mode='w') as f:
        f.write(dataframe.to_csv(index=False))
    print('CSV file write sucessfull')
    return True

#Read and wirte text files using tensorflow to and form google cloud storage bucket
def gcs_write_text_file(filepath,stringdata, opts=None):
    with tf.io.gfile.GFile(filepath, mode='w') as f:
        f.write(stringdata)
        print('text file write sucessfull')
    return True

#gcs read and write the trained models 
def gcs_write_trained_model(filepath, model):
    with tf.io.gfile.GFile(filepath+'/randomforestregressor.pkl', mode='wb') as f:
        pickle.dump(model,f)
        print('trained model write sucessful for gcs path')
    return True

#Read and write csv files from local storage
def local_read_csv_file(filepath):
    dataframe_read = pd.read_csv(filepath)
    print('\n\nREAD SUCESSFULL\nFROM: {}'.format(filepath))
    return dataframe_read

def local_write_as_csv_file(filepath, dataframe_write,metrics_string):
    dataframe_write.to_csv(filepath+'randomforestregression_output.csv', index=False)
    print('\n\nOUTPUT DATAFRAME WRITE SUCESSFULL\nTO: {}'.format(filepath))
    file = open(filepath+'random_forest_regression_metrics.txt','w')
    file.write(metrics_string)
    file.close()
    print('\n\nREGRESSION METRICS WRITE SUCESSFULL\nTO: {}'.format(filepath))
    
    
#Read and write the trained model from the local storage
def local_write_trained_model(filepath, model):
    file =open(str(filepath)+'/model.pkl', 'wb')
    pickle.dump(model, file)
    file.close()
    print('SAVED THE MODEL SUCESSFULLY TO PATH:\n',filepath)

def local_read_trained_model(filepath):
    file = open(filepath, 'rb')
    obj = pickle.load(file)
    file.close()
    print('MODEL LOADED SUCESSFULLY FROM PATH:\n',filepath)
    return obj
