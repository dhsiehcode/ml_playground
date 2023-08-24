import numpy as np
import pandas as pd
import tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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


def clean_and_get_data(record, outcome):

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


    ## Encoe categorical variables

    #result = pd.get_dummies(result,columns=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                                            #'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
                                            #'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS'], dtype=int)

    #result


    # drop ID if necessary
    result.drop(columns=['ID'], inplace=True)

    y = result['OUTCOME']
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    #exit(0)
    X = result.iloc[:, :(result.shape[1] - 1)]

    income_categorical, income_bins = pd.cut(x = X['AMT_INCOME_TOTAL'], bins = 10, labels=False, retbins=True)
    birth_categorical, birth_bins = pd.cut(x = X['DAYS_BIRTH'], bins = 10, labels=False, retbins=True)
    employed_categorical, employed_bins = pd.cut(x = X['DAYS_EMPLOYED'], bins = 10, labels=False, retbins=True)

    X['AMT_INCOME_TOTAL'] = income_categorical
    X['DAYS_BIRTH'] = birth_categorical
    X['DAYS_EMPLOYED'] = employed_categorical

    oe = OrdinalEncoder()

    for col in X.columns:
        X[col] = oe.fit_transform(X[col].to_numpy(copy=True).reshape(-1,1)).reshape(-1)
    #X['AMT_INCOME_TOTAL'] = oe.fit_transform(X['AMT_INCOME_TOTAL'].to_numpy(copy = True).reshape(-1,1)).reshape(-1)
    #X['DAYS_BIRTH'] = oe.fit_transform(X['DAYS_BIRTH'].to_numpy(copy = True).reshape(-1, 1)).reshape(-1)
    #X['DAYS_EMPLOYED'] = oe.fit_transform(X['DAYS_EMPLOYED'].to_numpy(copy = True).reshape(-1, 1)).reshape(-1)

    #print(X.head(5))


    return train_test_split(X, y, train_size=0.7, test_size=0.3)


def data_visualization(df, label):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)

    df = df.join(label, on = 'ID')

    df.rename(columns={0:"OUTCOME"}, inplace=True)

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

    id_birth = df[['ID', 'DAYS_BIRTH', 'OUTCOME']]
    id_birth.sort_index(inplace=True)
    cpy = id_birth.copy()
    cpy['DAYS_BIRTH'] = cpy['DAYS_BIRTH'].apply(lambda x:-1 * x / 365)
    for outcome in cpy['OUTCOME'].unique():
        ax1.scatter(cpy.loc[cpy['OUTCOME'] == outcome, 'ID'], cpy.loc[cpy['OUTCOME'] == outcome, 'DAYS_BIRTH'])
    ax1.set_title('Age')
    ax1.xlabel('ID number')
    ax1.ylabel('Age in years')


    id_employed = df[['ID', 'DAYS_EMPLOYED', 'OUTCOME']]
    id_employed = id_employed[id_employed['DAYS_EMPLOYED'] < 0]
    id_employed.sort_index(inplace=True)
    cpy = id_employed.copy()
    cpy['DAYS_EMPLOYED'] = cpy['DAYS_EMPLOYED'].apply(lambda x:-1 * x)
    for outcome in cpy['OUTCOME'].unique():
        ax2.scatter(cpy.loc[cpy['OUTCOME'] == outcome ,'ID'], cpy.loc[cpy['OUTCOME'] == outcome, 'DAYS_EMPLOYED'])
    ax2.set_title('Days Employed')
    ax2.xlabel('ID number')
    ax2.ylabel('Days employed')

    id_unemployed = df[['ID', 'DAYS_EMPLOYED', 'OUTCOME']]
    id_unemployed = id_unemployed[id_unemployed['DAYS_EMPLOYED'] > 0]
    id_unemployed.sort_index(inplace=True)
    cpy = id_unemployed.copy()
    cpy['DAYS_EMPLOYED'] = cpy['DAYS_EMPLOYED'].apply(lambda x:-1 * x)
    for outcome in cpy['OUTCOME'].unique():
        ax3.scatter(cpy.loc[cpy['OUTCOME'] == outcome, 'ID'], cpy.loc[cpy['OUTCOME'] == outcome, 'DAYS_EMPLOYED'])
    ax3.set_title('Days Unemployed')
    ax3.xlabel('ID number')
    ax3.ylabel('Days unemployed')

    id_income = df[['ID', 'AMT_INCOME_TOTAL', 'OUTCOME']]
    id_income.sort_index(inplace=True)
    for outcome in id_income['OUTCOME'].unique():
        ax4.scatter(id_income.loc[id_income['OUTCOME'] == outcome, 'ID'], id_income.loc[id_income['OUTCOME'] == outcome, 'AMT_INCOME_TOTAL'])
    ax4.set_title('Income')
    ax4.xlabel('ID number')
    ax4.ylabel('Income')

    plt.tight_layout()

    fig.savefig('credit_score_plots')

    ## some fancier stuff

    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)




def NBC(X_train, X_test, y_train, y_test):



    gNBC = naive_bayes.GaussianNB()



    gNBC.fit(X_train, y_train)

    print("Gaussian NBC")

    print((gNBC.predict(X_test[:5])))

    print(y_test[:5])

    print(gNBC.score(X_test, y_test))

    bNBC = naive_bayes.BernoulliNB()

    bNBC.fit(X_train, y_train)

    print("Bernoulli NBC")

    print(bNBC.predict(X_test[:5]))

    print(y_test[:5])

    print(bNBC.score(X_test, y_test))





    X_train_categorical = X_train[['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']]
    X_test_categorical = X_test[['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']]

    categorical_NBC = naive_bayes.CategoricalNB()

    categorical_NBC.fit(X_train_categorical, y_train)

    print("Categorical NBC (only 3 features")

    print(categorical_NBC.predict(X_test_categorical[:5]))
    print(y_test[:5])

    print(categorical_NBC.score(X_test_categorical, y_test))



def DT(X_train, X_test, y_train, y_test):

    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]


    params_dict = {'criterion:': ['gini', 'entropy'],
                   #'max_depth': [None, ],
                   'min_sample_splits': [2, 4, 8, 16, 32, 64, 128],
                   'max_features': [None, num_features / 2, num_features / 3, num_features / 4]}

    dt = DecisionTreeClassifier()

    random_search = RandomizedSearchCV(dt,
                                       params_dict,
                                       verbose=1,
                                       return_train_score=True)

    random_search.fit(X_train, y_train)


    print(f"DT Acc:{random_search.score(X_test, y_test)}")




if __name__ == '__main__':

    outcome = tools.import_data(applicant_outcome)

    record = tools.import_data(applicant_record)

    y_label = get_y(outcome)

    X_train, X_test, y_train, y_test = clean_and_get_data(record,y_label)

    #exit(0)

    #NBC(X_train, X_test, y_train, y_test)

    data_visualization(record, y_label)



    exit(0)











































