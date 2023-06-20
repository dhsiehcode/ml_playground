import numpy as np
import pandas as pd
import tools
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

applicant_record = 'C:\Dennis\Personal\Projects\ml_playground\data\credit_score\\application_record.csv'

applicant_outcome = 'C:\Dennis\Personal\Projects\ml_playground\data\credit_score\\credit_record.csv'

def get_y(df):



    # sort the IDS first

    unique_id = pd.unique(df['ID'])

    y_labels = {}

    y_labels_np = np.empty((unique_id.shape[0], 2))

    loc = 0

    for id in unique_id:
        temp = df[df['ID'] == id]

        result = temp['STATUS'].str.contains('^[0-9]$', regex = True)

        result = result[result == True]

        if result.sum() > 0:
            y_labels[id] = 'bad'

        else:
            y_labels[id] = 'good'

    return pd.DataFrame.from_dict(y_labels, orient='index')

# get rid of rows that don't exist

def clean_data(record, outcome):

    result = record.join(outcome, on = 'ID')

    result.rename(columns={0:"OUTCOME"}, inplace=True)

    #print(result)

    result.dropna(axis = 0, how='any', subset=['OUTCOME'])


    # drop ID if necessary
    result.drop(labels='ID')

    return result


def get_data(df):


    y = df['OUTCOME']
    X = df[:, :df.shape[1] - 1]

    return train_test_split(X, y, train_size=0.7, test_size=0.3)



if __name__ == '__main__':

    outcome = tools.import_data(applicant_outcome)

    record = tools.import_data(applicant_record)

    y_label = get_y(outcome)

    combined = clean_data(record,y_label)

    exit(0)











































