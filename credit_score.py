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

    result.fillna(axis=0, value={'OCCUPATION_TYPE':'Unknown'}, inplace=True)

    ## Convert to discrete (such as age and employment) if necessary

    convert_to_discrete = False

    if (convert_to_discrete):
        result['DAYS_BIRTH'] = -1 * result['DAYS_BIRTH'] / 356

        # -1 Represents employed
        result[result['DAYS_EMPLOYED'] < 0] = -1
        # 1 Represenets unemployed
        result[result['DAYS_EMPLOYED'] > 0] = 1
        # Negative number represents years unemployed
        #result[result['DAYS_EMPLOYED'] > 0] = result[result['DAYS_EMPLOYED'] < 0] * -1 / 365

        # Divide salaries into 10 based on max and min





    # drop ID if necessary
    result.drop(labels='ID')

    return result


def get_data(df):


    y = df['OUTCOME']
    X = df[:, :df.shape[1] - 1]

    return train_test_split(X, y, train_size=0.7, test_size=0.3)


def data_visualization(df):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)

    # showing gender


    ax1.bar(x = df['CODE_GENDER'].count)
    ax1.set_title('Gender Bar Chart')

    ax2.bar(x = df['FLAG_OWN_CAR'].count)
    ax2.set_title('Owns Car')

    ax3.bar(x = df['FLAG_OWN_REALTY'].count)
    ax3.set_title('Owns Reality')

    ax4.bar(x = df['FLAG_MOBIL'].count)
    ax4.set_title('Provides mobile number')

    ax5.bar(x = df['FLAG_WORK_PHONE'].count)
    ax5.set_title('Provides work phone')

    ax6.bar(x = df['FLAG_PHONE'].count)
    ax6.set_title('Provides home phone')

    ax7.bar(x = df['FLAG_EMAIL'].count)
    ax7.set_title('Provides  email')

    fig.show()


if __name__ == '__main__':

    outcome = tools.import_data(applicant_outcome)

    record = tools.import_data(applicant_record)

    y_label = get_y(outcome)

    combined = clean_data(record,y_label)

    exit(0)











































