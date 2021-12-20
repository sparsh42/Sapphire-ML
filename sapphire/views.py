from typing import DefaultDict
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.templatetags.static import static
from django.core.files import File
import csv
import io

from pandas.io.parsers import read_csv
from sapphire.models import CSV_model
import pandas as pd
import numpy as np
from scipy.stats import stats
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.use('Agg')


def CheckTargetUniq(df, targetVar):
    # Checks if there are more than 1 unique variables in the target column
    if df[targetVar].nunique() > 1 and df[targetVar].nunique() < 11:
        return False
    else:
        return True


def dropColumn(df, arr):
    # Drops columns from a DataFrame, returns new df
    df = df.drop(arr, axis=1)
    return df


def getNullCols(df):
    # print(df.columns[df.isnull().any()].tolist())
    return df.columns[df.isnull().any()].tolist()


def getColNulls(df):
    # See null values in all features
    total_nulls = []
    for item in df:
        if df[item].isnull().sum() > 0:
            total_nulls.append(df[item].isnull().sum())
        else:
            continue
    return total_nulls


def replaceColNulls(df, col, val):
    # Replace null values in a column with given value
    # print('val')
    # print(val)
    df[col] = df[col].replace(np.nan, val)
    return df


def dropNullRow(df, col):
    # Drop null values in a column with given value
    arr = df[df[col].isnull()].index.tolist()
    # print(arr)
    df = df.drop(arr)
    return df


def targetVarType(df, targetVar):
    # See if the vars are categorical or numerical
    if df[targetVar].dtype == 'int64':
        return 0
    elif df[targetVar].dtype == 'float64':
        return 1
    else:
        return 2


def Zscore_outlier(df, col_name):
    if df[col_name].dtype == 'int64' or df[col_name].dtype == 'float64':
        df = df[col_name]
        m = np.mean(df)
        sd = np.std(df)
        out = []
        indices = []
        b_list = [b(x, m, sd) for x in df]
        out.append(b_list)
        out = out[0]
        for i in range(len(out)):
            if out[i] == True:
                indices.append(i)
        out = [out.remove(x) for x in out if type(x) == 'Bool']
        return indices
    else:
        return []


def b(x, m, sd):
    z = (x-m)/sd
    if np.abs(z) > 3:
        return True
    else:
        return False


def mostFreq(df, col_name):
    # For Categorical Data -- Get most frequent value
    most_freq = list(mode(df[col_name]))
    return most_freq[0][0]


def getMean(df, col_name):
    # For numerical data -- Get the mean
    return df[col_name].mean(skipna=True)


def getMedian(df, col_name):
    # For numerical data -- Get the mean
    return df[col_name].median(skipna=True)


def getMode(df, col_name):
    return df[col_name].mode()[0]

    # -------------- Label Encoding -----------


def encodeLabels(df):
    for item in df:
        if df[item].dtype != 'int64' and df[item].dtype != 'float64':
            label_encoder = preprocessing.LabelEncoder()
            # Encode labels in column 'species'.
            df[item] = label_encoder.fit_transform(df[item])
    return df

# ----------- Data Splitting and Modeling --------


def train_test_splitting(df, targetVar, percent):
    X = df.drop(targetVar, axis=1)
    y = df[targetVar].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=percent/100, random_state=0)
    print("train_test_splitting: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    return standardScaler(X_train, X_test, y_train, y_test, df, targetVar)


def standardScaler(X_train, X_test, y_train, y_test, df, targetVar):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print("standardScaler: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    return runModels(X_train, X_test, y_train, y_test, df, targetVar)


def runModels(X_train, X_test, y_train, y_test, df, targetVar):
    print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    SVC_score = SVC_(X_train, X_test, y_train, y_test)
    print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    RFC_score = RFC(X_train, X_test, y_train, y_test)
    print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    DTC_score = DTC(X_train, X_test, y_train, y_test)
    print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    LR_score = LR(X_train, X_test, y_train, y_test)
    print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    KNN_score = KNN(X_train, X_test, y_train, y_test)
    print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    a = targetVar
    if list(df[a].unique()) == list(range(0, int(df[a].nunique()))):
        XGB_score = XGB(X_train, X_test, y_train, y_test)
        print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        return (SVC_score, RFC_score, DTC_score, LR_score, KNN_score, XGB_score)
    # ,  XGB_score)
    return (SVC_score, RFC_score, DTC_score, LR_score, KNN_score)


def SVC_(X_train, X_test, y_train, y_test):
    scores = []
    classifier = SVC(kernel='rbf', random_state=0)
    # Fitting the model
    classifier.fit(X_train, y_train)
    # Predicting the models
    y_pred = classifier.predict(X_test)
    scores.append(int(accuracy_score(y_test, y_pred)*10000)/100)
    scores.append(int(recall_score(y_test, y_pred, average='macro')*10000)/100)
    scores.append(
        int(precision_score(y_test, y_pred, average='macro')*10000)/100)
    return scores


def DTC(X_train, X_test, y_train, y_test):
    scores = []
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    # Fitting the model
    classifier.fit(X_train, y_train)
    # Predicting the models
    y_pred = classifier.predict(X_test)
    scores.append(int(accuracy_score(y_test, y_pred)*10000)/100)
    scores.append(int(recall_score(y_test, y_pred, average='macro')*10000)/100)
    scores.append(
        int(precision_score(y_test, y_pred, average='macro')*10000)/100)
    return scores


def RFC(X_train, X_test, y_train, y_test):
    scores = []
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    # Fitting the model
    classifier.fit(X_train, y_train)
    # Predicting the models
    y_pred = classifier.predict(X_test)
    scores.append(int(accuracy_score(y_test, y_pred)*10000)/100)
    scores.append(int(recall_score(y_test, y_pred, average='macro')*10000)/100)
    scores.append(
        int(precision_score(y_test, y_pred, average='macro')*10000)/100)
    return scores


def XGB(X_train, X_test, y_train, y_test):
    scores = []
    classifier = XGBClassifier(use_label_encoder=False, verbosity=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    scores.append(int(accuracy_score(y_test, y_pred)*10000)/100)
    scores.append(int(recall_score(y_test, y_pred, average='macro')*10000)/100)
    scores.append(
        int(precision_score(y_test, y_pred, average='macro')*10000)/100)
    return scores


def LR(X_train, X_test, y_train, y_test):
    scores = []
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    # Predicting the models
    y_pred = classifier.predict(X_test)
    scores.append(int(accuracy_score(y_test, y_pred)*10000)/100)
    scores.append(int(recall_score(y_test, y_pred, average='macro')*10000)/100)
    scores.append(
        int(precision_score(y_test, y_pred, average='macro')*10000)/100)
    return scores


def KNN(X_train, X_test, y_train, y_test):
    scores = []
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    # Predicting the models
    y_pred = classifier.predict(X_test)
    scores.append(int(accuracy_score(y_test, y_pred)*10000)/100)
    scores.append(int(recall_score(y_test, y_pred, average='macro')*10000)/100)
    scores.append(
        int(precision_score(y_test, y_pred, average='macro')*10000)/100)
    return scores


def home(request):

    if request.method == "POST" and request.POST.get('sample', False):

        csv_name = request.session.get("csv", default=False)
        if csv_name:
            if CSV_model.objects.filter(csv_name=csv_name):
                csv_f = CSV_model.objects.filter(csv_name=csv_name)[0]
                csv_f.delete()
            del request.session["csv"]

        file_path = 'https://sapphire-ml.herokuapp.com' + \
            static('csv/water_potability.csv')
        df = read_csv(file_path)

        req_csv = ContentFile(df.to_csv(index=False))
        req_csv.name = "Anything.csv"

        csv_model = CSV_model(csv_file=req_csv)
        csv_model.save()

        request.session["csv"] = csv_model.getfilename()

        csv_model.csv_name = csv_model.getfilename()
        csv_model.save()

        return HttpResponseRedirect(reverse('tvar'))

    if not request.session.get("csv", default=False):
        if request.method == "POST":
            try:
                if request.FILES['csvfile']:
                    print("CSV File Uploaded")
                    req_csv = request.FILES['csvfile']
                    # print(req_csv)
                    # print(req_csv.size/1024)

                    if not req_csv.name.endswith('.csv'):
                        error = 'Please upload a CSV file'
                        # messages.error(request, 'THIS IS NOT A CSV FILE')
                        return render(request, 'index.html', {'title': 'home', 'error': error})

                    if req_csv.size/1024 > 2048:
                        error = 'Please upload a CSV file of 2MB or less in size'
                        # messages.error(request, 'THIS IS NOT A CSV FILE')
                        return render(request, 'index.html', {'title': 'home', 'error': error})

                    csv_model = CSV_model(csv_file=req_csv)
                    csv_model.save()

                    request.session["csv"] = csv_model.getfilename()
                    # print("Working!!!")

                    csv_model.csv_name = csv_model.getfilename()
                    csv_model.save()

                    return HttpResponseRedirect(reverse('home'))
            except:
                error = "There was something wrong"
        return render(request, 'index.html', {'title': 'home'})
    else:
        return HttpResponseRedirect(reverse('tvar'))

        csv_name = request.session.get("csv", default=False)
        csv_f = CSV_model.objects.filter(csv_name=csv_name)[0]
        # print(csv_name)

        if csv_f:

            df = pd.read_csv(csv_f.csv_file.path)
            # print(df)

            # print(df.columns.tolist())
            columns = df.columns.tolist()

            if csv_f.target_variable:
                t_var = csv_f.target_variable

                columns.remove(t_var)

                if not df.isnull().sum().all():
                    isnull = getNullCols(df)
                    isnull_type = []
                    isnull_count = getColNulls(df)

                    for col in isnull:
                        isnull_type.append(targetVarType(df, col))

                drop_column = csv_f.drop_column
                outlier_val = csv_f.outlier
                heatmap_val = csv_f.heatmap
                if not outlier_val:
                    if not csv_f.drop_column:
                        if request.method == "POST":
                            if not df.isnull().sum().all():
                                for i, col in enumerate(isnull):
                                    null_choice = request.POST.get(
                                        'null-row' + str(i), False)
                                    # print('null_choice - ' + str(i))
                                    # print(null_choice)
                                    if isnull_type[i] == 0:
                                        if null_choice == "mode":
                                            df = replaceColNulls(
                                                df, col, getMode(df, col))
                                        elif null_choice == "mean":
                                            df = replaceColNulls(
                                                df, col, getMean(df, col))
                                        elif null_choice == "median":
                                            df = replaceColNulls(
                                                df, col, getMedian(df, col))
                                        elif null_choice == "drop":
                                            df = dropNullRow(df, col)
                                        else:
                                            error = "There was something wrong"
                                    elif isnull_type[i] == 1:
                                        if null_choice == "mean":
                                            df = replaceColNulls(
                                                df, col, getMean(df, col))
                                        elif null_choice == "drop":
                                            df = dropNullRow(df, col)
                                        else:
                                            error = "There was something wrong"
                                    else:
                                        if null_choice == "mode":
                                            df = replaceColNulls(
                                                df, col, mostFreq(df, col))
                                        elif null_choice == "drop":
                                            df = dropNullRow(df, col)
                                        else:
                                            error = "There was something wrong"

                                    if df.isnull().sum().all():
                                        break

                            if request.POST.getlist('drop-column[]', False):
                                drop_columns = request.POST.getlist(
                                    'drop-column[]')
                                for col in drop_columns:
                                    if not col in columns:
                                        error = "There was something wrong"

                                df = dropColumn(df, drop_columns)
                                # print(df.to_string())

                            new_df = df.to_csv(index=False)

                            csv_f.deleteCSVfile()

                            updated_file = ContentFile(new_df)
                            updated_file.name = csv_f.getfilename()

                            csv_f.csv_file = updated_file
                            csv_f.drop_column = True
                            csv_f.save()
                            return HttpResponseRedirect(reverse('home'))

                        return render(request, 'index.html', {'title': 'home', 'headers': columns, 't_var': t_var, 'isnull': zip(isnull, isnull_type, isnull_count), 'isnull_count': len(isnull)})
                    else:
                        outliers = []
                        outlier_type = []
                        outlier_count = []
                        # print('z values')
                        for col in columns:
                            z = Zscore_outlier(df, col)
                            if z != []:
                                # print(z)
                                outliers.append(col)
                                outlier_type.append(targetVarType(df, col))
                                outlier_count.append(len(z))
                        if outliers != []:
                            if request.method == "POST":
                                for i, col in enumerate(outliers):
                                    outlier_choice = request.POST.get(
                                        'outlier-row' + str(i), False)

                                    if outlier_type[i] == 0:
                                        if outlier_choice == "mode":
                                            df = replaceColNulls(
                                                df, col, getMode(df, col))
                                        elif outlier_choice == "mean":
                                            df = replaceColNulls(
                                                df, col, getMean(df, col))
                                        elif outlier_choice == "median":
                                            df = replaceColNulls(
                                                df, col, getMedian(df, col))
                                        elif outlier_choice == "drop":
                                            df = dropNullRow(df, col)
                                        else:
                                            error = "There was something wrong"
                                    elif outlier_type[i] == 1:
                                        if outlier_choice == "mean":
                                            df = replaceColNulls(
                                                df, col, getMean(df, col))
                                        elif outlier_choice == "drop":
                                            df = dropNullRow(df, col)
                                        else:
                                            error = "There was something wrong"
                                    else:
                                        if outlier_choice == "mode":
                                            df = replaceColNulls(
                                                df, col, mostFreq(df, col))
                                        elif outlier_choice == "drop":
                                            df = dropNullRow(df, col)
                                        else:
                                            error = "There was something wrong"
                                csv_f.outlier = True
                                csv_f.save()
                                return HttpResponseRedirect(reverse('home'))
                        else:
                            csv_f.outlier = True
                            csv_f.save()
                            return HttpResponseRedirect(reverse('home'))

                        return render(request, 'index.html', {'title': 'home', 'headers': columns, 't_var': t_var, 'drop_column': drop_column, 'outliers_val': zip(outliers, outlier_type, outlier_count)})
                else:
                    if heatmap_val:
                        df = encodeLabels(df)
                        final_val = train_test_splitting(
                            df, csv_f.target_variable, csv_f.train_test)

                        return render(request, 'index.html', {'title': 'home', 'headers': columns, 't_var': t_var, 'outlier': outlier_val, 'heatmap': heatmap_val, 'final': final_val})
                    else:
                        plt.subplots(figsize=(10, 8))
                        h_map = sns.heatmap(df.corr(), annot=True)
                        h_map.figure.savefig(
                            "media/heatmap/" + csv_name.split('.')[0] + ".png")
                        temp_path = "heatmap/" + \
                            csv_name.split('.')[0] + ".png"

                        if request.method == "POST":
                            csv_f.heatmap = temp_path
                            csv_f.train_test = request.POST.get(
                                'train-test', False)
                            if request.POST.getlist('drop-column[]', False):
                                drop_columns = request.POST.getlist(
                                    'drop-column[]')
                                for col in drop_columns:
                                    if not col in columns:
                                        error = "There was something wrong"

                                df = dropColumn(df, drop_columns)

                                new_df = df.to_csv(index=False)
                                csv_f.deleteCSVfile()

                                updated_file = ContentFile(new_df)
                                updated_file.name = csv_f.getfilename()

                                csv_f.csv_file = updated_file
                                print(datetime.datetime.now())
                            csv_f.save()
                            return HttpResponseRedirect(reverse('home'))
                    return render(request, 'index.html', {'title': 'home', 'headers': columns, 't_var': t_var, 'outlier': outlier_val, 'heatmap_temp': temp_path})
            else:
                if request.method == "POST":
                    try:
                        if request.POST.get('target-variable', False):
                            t_variable = request.POST.get('target-variable')
                            if t_variable in columns:

                                if CheckTargetUniq(df, t_variable):
                                    error = "Cannot classify more than 10 class labels or a single class label. Please use another CSV."
                                    csv_name = request.session.get(
                                        "csv", default=False)
                                    if csv_name and CSV_model.objects.filter(csv_name=csv_name):
                                        csv_f = CSV_model.objects.filter(
                                            csv_name=csv_name)[0]
                                        csv_f.delete()
                                        del request.session["csv"]
                                    elif csv_name:
                                        del request.session["csv"]
                                    return render(request, 'index.html', {'title': 'home', 'error': error})

                                csv_f.target_variable = t_variable
                                csv_f.train_test = request.POST.get(
                                    'train-test')
                                csv_f.save()
                            else:
                                error = "Target Variable is not present in CSV Header."
                                return render(request, 'index.html', {'title': 'home', 'error': error})

                            return HttpResponseRedirect(reverse('home'))
                    except:
                        error = "There was something wrong"
                return render(request, 'index.html', {'title': 'home', 'headers': columns})
        return render(request, 'index.html', {'title': 'home'})


def tvar(request):

    csv_name = request.session.get("csv", default=False)
    csv_f = CSV_model.objects.filter(csv_name=csv_name)[0]

    if csv_f:

        df = pd.read_csv(csv_f.csv_file.path)
        columns = df.columns.tolist()
        if csv_f.target_variable:
            return HttpResponseRedirect(reverse('column'))
        else:
            if request.method == "POST":
                try:
                    if request.POST.get('target-variable', False):
                        t_variable = request.POST.get('target-variable')
                        if t_variable in columns:

                            if CheckTargetUniq(df, t_variable):
                                error = "Cannot classify more than 10 class labels or a single class label. Please use another CSV."
                                csv_name = request.session.get(
                                    "csv", default=False)
                                if csv_name and CSV_model.objects.filter(csv_name=csv_name):
                                    csv_f = CSV_model.objects.filter(
                                        csv_name=csv_name)[0]
                                    csv_f.delete()
                                    del request.session["csv"]
                                elif csv_name:
                                    del request.session["csv"]
                                return render(request, 'target-variable.html', {'title': 'tvar', 'error': error})

                            csv_f.target_variable = t_variable
                            csv_f.train_test = request.POST.get(
                                'train-test')
                            csv_f.save()
                        else:
                            error = "Target Variable is not present in CSV Header."
                            return render(request, 'target-variable.html', {'title': 'tvar', 'error': error})

                        return HttpResponseRedirect(reverse('column'))
                except:
                    error = "There was something wrong"
        # return render(request, 'index.html', {'title': 'home', 'headers': columns})
        return render(request, 'target-variable.html', {'title': 'tvar', 'headers': columns})
    else:
        return HttpResponseRedirect(reverse('home'))


def column(request):
    csv_name = request.session.get("csv", default=False)
    csv_f = CSV_model.objects.filter(csv_name=csv_name)[0]

    if csv_f:
        df = pd.read_csv(csv_f.csv_file.path)
        # print(df)

        # print(df.columns.tolist())
        columns = df.columns.tolist()

        if csv_f.target_variable:
            t_var = csv_f.target_variable

            columns.remove(t_var)

            if not df.isnull().sum().all():
                isnull = getNullCols(df)
                isnull_type = []
                isnull_count = getColNulls(df)

                for col in isnull:
                    isnull_type.append(targetVarType(df, col))

            if not csv_f.drop_column:
                if request.method == "POST":
                    if not df.isnull().sum().all():
                        for i, col in enumerate(isnull):
                            null_choice = request.POST.get(
                                'null-row' + str(i), False)
                            # print('null_choice - ' + str(i))
                            # print(null_choice)
                            if isnull_type[i] == 0:
                                if null_choice == "mode":
                                    df = replaceColNulls(
                                        df, col, getMode(df, col))
                                elif null_choice == "mean":
                                    df = replaceColNulls(
                                        df, col, getMean(df, col))
                                elif null_choice == "median":
                                    df = replaceColNulls(
                                        df, col, getMedian(df, col))
                                elif null_choice == "drop":
                                    df = dropNullRow(df, col)
                                else:
                                    error = "There was something wrong"
                            elif isnull_type[i] == 1:
                                if null_choice == "mean":
                                    df = replaceColNulls(
                                        df, col, getMean(df, col))
                                elif null_choice == "drop":
                                    df = dropNullRow(df, col)
                                else:
                                    error = "There was something wrong"
                            else:
                                if null_choice == "mode":
                                    df = replaceColNulls(
                                        df, col, mostFreq(df, col))
                                elif null_choice == "drop":
                                    df = dropNullRow(df, col)
                                else:
                                    error = "There was something wrong"

                            if df.isnull().sum().all():
                                break

                    if request.POST.getlist('drop-column[]', False):
                        drop_columns = request.POST.getlist(
                            'drop-column[]')
                        for col in drop_columns:
                            if not col in columns:
                                error = "There was something wrong"

                        df = dropColumn(df, drop_columns)
                        # print(df.to_string())

                    new_df = df.to_csv(index=False)

                    csv_f.deleteCSVfile()

                    updated_file = ContentFile(new_df)
                    updated_file.name = csv_f.getfilename()

                    csv_f.csv_file = updated_file
                    csv_f.drop_column = True
                    csv_f.save()
                    return HttpResponseRedirect(reverse('home'))

                return render(request, 'null-column.html', {'title': 'home', 'headers': columns, 't_var': t_var, 'isnull': zip(isnull, isnull_type, isnull_count), 'isnull_count': len(isnull)})
            else:
                return HttpResponseRedirect(reverse('outlier'))
        else:
            return HttpResponseRedirect(reverse('tvar'))
    else:
        return HttpResponseRedirect(reverse('home'))


def outlier(request):
    csv_name = request.session.get("csv", default=False)
    csv_f = CSV_model.objects.filter(csv_name=csv_name)[0]

    if csv_f:
        df = pd.read_csv(csv_f.csv_file.path)
        # print(df)

        # print(df.columns.tolist())
        columns = df.columns.tolist()

        if csv_f.target_variable:
            t_var = csv_f.target_variable

            columns.remove(t_var)

            if not df.isnull().sum().all():
                isnull = getNullCols(df)
                isnull_type = []
                isnull_count = getColNulls(df)

                for col in isnull:
                    isnull_type.append(targetVarType(df, col))

            drop_column = csv_f.drop_column
            outlier_val = csv_f.outlier
            heatmap_val = csv_f.heatmap

            if not outlier_val:
                outliers = []
                outlier_type = []
                outlier_count = []
                # print('z values')
                for col in columns:
                    z = Zscore_outlier(df, col)
                    if z != []:
                        # print(z)
                        outliers.append(col)
                        outlier_type.append(targetVarType(df, col))
                        outlier_count.append(len(z))
                if outliers != []:
                    if request.method == "POST":
                        for i, col in enumerate(outliers):
                            outlier_choice = request.POST.get(
                                'outlier-row' + str(i), False)

                            if outlier_type[i] == 0:
                                if outlier_choice == "mode":
                                    df = replaceColNulls(
                                        df, col, getMode(df, col))
                                elif outlier_choice == "mean":
                                    df = replaceColNulls(
                                        df, col, getMean(df, col))
                                elif outlier_choice == "median":
                                    df = replaceColNulls(
                                        df, col, getMedian(df, col))
                                elif outlier_choice == "drop":
                                    df = dropNullRow(df, col)
                                else:
                                    error = "There was something wrong"
                            elif outlier_type[i] == 1:
                                if outlier_choice == "mean":
                                    df = replaceColNulls(
                                        df, col, getMean(df, col))
                                elif outlier_choice == "drop":
                                    df = dropNullRow(df, col)
                                else:
                                    error = "There was something wrong"
                            else:
                                if outlier_choice == "mode":
                                    df = replaceColNulls(
                                        df, col, mostFreq(df, col))
                                elif outlier_choice == "drop":
                                    df = dropNullRow(df, col)
                                else:
                                    error = "There was something wrong"
                        csv_f.outlier = True
                        csv_f.save()
                        return HttpResponseRedirect(reverse('split'))
                else:
                    csv_f.outlier = True
                    csv_f.save()
                    return HttpResponseRedirect(reverse('split'))

                return render(request, 'outlier.html', {'title': 'outlier', 'headers': columns, 't_var': t_var, 'drop_column': drop_column, 'outliers_val': zip(outliers, outlier_type, outlier_count)})
            # else:
            #     return HttpResponseRedirect(reverse('split'))
            else:
                return HttpResponseRedirect(reverse('split'))
        else:
            return HttpResponseRedirect(reverse('tvar'))
    else:
        return HttpResponseRedirect(reverse('home'))


def split(request):
    csv_name = request.session.get("csv", default=False)
    csv_f = CSV_model.objects.filter(csv_name=csv_name)[0]

    if csv_f:
        df = pd.read_csv(csv_f.csv_file.path)
        # print(df)

        # print(df.columns.tolist())
        columns = df.columns.tolist()

        if csv_f.target_variable:
            t_var = csv_f.target_variable

            columns.remove(t_var)
            drop_column = csv_f.drop_column
            outlier_val = csv_f.outlier
            heatmap_val = csv_f.heatmap

            if not drop_column:
                return HttpResponseRedirect(reverse('column'))
            if not outlier_val:
                return HttpResponseRedirect(reverse('outlier'))

            if not heatmap_val:
                plt.subplots(figsize=(10, 8))
                h_map = sns.heatmap(df.corr(), annot=True)
                h_map.figure.savefig(
                    "media/heatmap/" + csv_name.split('.')[0] + ".png")
                temp_path = "heatmap/" + \
                    csv_name.split('.')[0] + ".png"

                if request.method == "POST":
                    csv_f.heatmap = temp_path
                    csv_f.train_test = request.POST.get(
                        'train-test', False)
                    if request.POST.getlist('drop-column[]', False):
                        drop_columns = request.POST.getlist(
                            'drop-column[]')
                        for col in drop_columns:
                            if not col in columns:
                                error = "There was something wrong"

                        df = dropColumn(df, drop_columns)

                        new_df = df.to_csv(index=False)
                        csv_f.deleteCSVfile()

                        updated_file = ContentFile(new_df)
                        updated_file.name = csv_f.getfilename()

                        csv_f.csv_file = updated_file
                        print(datetime.datetime.now())
                    csv_f.save()
                    return HttpResponseRedirect(reverse('predictions'))
                return render(request, 'train-test-split.html', {'split': 'home', 'headers': columns, 't_var': t_var, 'outlier': outlier_val, 'heatmap_temp': temp_path})
            else:
                return HttpResponseRedirect(reverse('predictions'))
        else:
            return HttpResponseRedirect(reverse('tvar'))
    else:
        return HttpResponseRedirect(reverse('home'))


def predictions(request):
    csv_name = request.session.get("csv", default=False)
    csv_f = CSV_model.objects.filter(csv_name=csv_name)[0]
    if csv_f:
        df = pd.read_csv(csv_f.csv_file.path)
        # print(df)

        # print(df.columns.tolist())
        columns = df.columns.tolist()
        if csv_f.target_variable:
            t_var = csv_f.target_variable

            columns.remove(t_var)
            drop_column = csv_f.drop_column
            outlier_val = csv_f.outlier
            heatmap_val = csv_f.heatmap

            if not drop_column:
                return HttpResponseRedirect(reverse('column'))
            if not outlier_val:
                return HttpResponseRedirect(reverse('outlier'))
            if not heatmap_val:
                return HttpResponseRedirect(reverse('split'))

            df = encodeLabels(df)
            final_val = train_test_splitting(
                df, csv_f.target_variable, csv_f.train_test)
                
            print("final_val: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

            return render(request, 'final-predictions.html', {'title': 'predictions', 'headers': columns, 'final': final_val})
            
        else:
            return HttpResponseRedirect(reverse('tvar'))
    else:
        return HttpResponseRedirect(reverse('home'))


def reset(request):
    csv_name = request.session.get("csv", default=False)
    if csv_name and CSV_model.objects.filter(csv_name=csv_name):
        csv_f = CSV_model.objects.filter(csv_name=csv_name)[0]
        csv_f.delete()
        del request.session["csv"]
    elif csv_name:
        del request.session["csv"]
    return HttpResponseRedirect(reverse('home'))


def about(request):
    return render(request, 'about.html', {'title': 'about'})


def page404(request):
    return render(request, '404.html', {'title': 'page404'})


def sample(request):
    return render(request, 'sample.html', {'title': 'sample'})
