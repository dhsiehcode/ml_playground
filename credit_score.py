import numpy as np
import pandas as pd
import tools
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
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

    ## drops all other nan

    result.dropna(axis=0, how='any', inplace=True)

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
    result.drop(columns=['ID'])

    return result


def get_data(df):


    y = df['OUTCOME']
    X = df.iloc[:, :(df.shape[1] - 1)]

    return train_test_split(X, y, train_size=0.7, test_size=0.3)


def data_visualization(df):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)

    # showing gender



    ax1.bar(x = df['CODE_GENDER'].value_counts().keys().to_numpy(), height = df['CODE_GENDER'].value_counts())
    ax1.set_title('Gender Bar Chart')

    ax2.bar(x = df['FLAG_OWN_CAR'].value_counts().keys(), height = df['FLAG_OWN_CAR'].value_counts())
    ax2.set_title('Owns Car')

    ax3.bar(x = df['FLAG_OWN_REALTY'].value_counts().keys(), height = df['FLAG_OWN_CAR'].value_counts())
    ax3.set_title('Owns Reality')

    ax4.bar(x = df['FLAG_MOBIL'].value_counts().keys().to_numpy(), height = df['FLAG_OWN_CAR'].value_counts())
    ax4.set_title('Provides mobile number')

    ax5.bar(x = df['FLAG_WORK_PHONE'].value_counts().keys(), height = df['FLAG_WORK_PHONE'].value_counts())
    ax5.set_title('Provides work phone')

    ax6.bar(x = df['FLAG_PHONE'].value_counts().keys(), height = df['FLAG_PHONE'].value_counts())
    ax6.set_title('Provides home phone')

    ax7.bar(x = df['FLAG_EMAIL'].value_counts().keys(), height = df['FLAG_EMAIL'].value_counts())
    ax7.set_title('Provides  email')

    ax8.bar(x = df['CNT_CHILDREN'].value_counts().keys(), height = df['CNT_CHILDREN'].value_counts())
    ax8.set_title('Number of Children')

    unemployed = df[df['DAYS_EMPLOYED'] > 0]
    employed = df[df['DAYS_EMPLOYED'] < 0]

    ax9.bar(x = ['Employed', 'Unemployed'], height = [employed.shape[0], unemployed.shape[0]])
    ax9.set_title('Employment')

    plt.tight_layout()

    fig.savefig('credit_score_bar_chart.png')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)

    id_birth = df[['ID', 'DAYS_BIRTH']]
    id_birth.sort_index(inplace=True)
    ax1.scatter(id_birth['ID'], id_birth['DAYS_BIRTH'])
    ax1.set_title('Birth Date')

    id_employed = df[[ 'ID', 'DAYS_EMPLOYED']]
    id_employed = id_employed[id_employed['DAYS_EMPLOYED'] < 0]
    id_employed.sort_index(inplace=True)
    ax2.scatter(id_employed['ID'], id_employed['DAYS_EMPLOYED'])
    ax2.set_title('Days Employed')

    id_unemployed = df[['ID', 'DAYS_EMPLOYED']]
    id_unemployed = id_unemployed[id_unemployed['DAYS_EMPLOYED'] > 0]
    id_unemployed.sort_index(inplace=True)
    ax3.scatter(id_unemployed['ID'], id_unemployed['DAYS_EMPLOYED'])
    ax4.set_title('Days Unemployed')

    id_income = df[['ID', 'AMT_INCOME_TOTAL']]
    id_income.sort_index(inplace=True)
    ax4.scatter(id_income['ID'], id_income['AMT_INCOME_TOTAL'])
    ax4.set_title('Income')

    plt.tight_layout()

    fig.savefig('credit_score_plots')


def NBC(X_train, X_test, y_train, y_test):

    NBC = GaussianNB()



    NBC.fit(X_train, y_train)

    print(NBC.score(X_test, y_test))


if __name__ == '__main__':

    outcome = tools.import_data(applicant_outcome)

    record = tools.import_data(applicant_record)

    y_label = get_y(outcome)

    combined = clean_data(record,y_label)

    X_train, X_test, y_train, y_test = get_data(combined)

    NBC(X_train, X_test, y_train, y_test)

    #data_visualization(record)



    exit(0)











































