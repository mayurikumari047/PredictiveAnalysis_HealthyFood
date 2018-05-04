def spend_money_heathy_food():
    # coding: utf-8

    # # Analysis and prediction on likelihood of spending more money for healthy foods using the young-people-survey dataset

    # ## Setup: Importing common modules and machine learning libraries

    # In[1]:

    import pandas as pd
    import numpy as np
    # get_ipython().run_line_magic('matplotlib', 'inline')
    from matplotlib import pyplot as plt
    import seaborn as sns
    import math
    from collections import Counter, OrderedDict
    pd.set_option('display.max_columns', 150)
    plt.style.use('bmh')
    from IPython.display import display
    import copy
    import warnings
    warnings.filterwarnings("ignore")
    #from __future__ import print_function

    # import of machine learning libraries
    from sklearn.metrics import recall_score, precision_score
    from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, r2_score, classification_report
    from sklearn.preprocessing import Imputer
    from sklearn.ensemble import RandomForestClassifier as RFR
    from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
    from sklearn.svm import SVC, LinearSVC

    # ### Reading and loading data in pandas dataframe

    # In[2]:

    def read_data(file_path):
        return pd.read_csv(file_path)

    # In[3]:

    file_path = 'young-people-survey/responses.csv'
    yps_data = read_data(file_path)

    # ## Data analysis of the young-people-survey dataset with respect to spending habbits and health concerns

    # In[4]:

    var_of_interest = 'Spending on healthy eating'
    yps_data.head(2)

    # In[5]:

    yps_data.describe()

    # ### Mean for the feature 'Spending on healthy eating' is 3.55 which is bit more than average.

    # ### Check for missing values in the data

    # In[6]:

    def check_null_values_in_data():
        null_fields = yps_data.isnull().sum().sort_values(ascending=False)
        print("Plot of null values for all the features in the data set.")
        null_fields.plot(kind='bar', figsize=(25, 5))
        plt.show()

    # In[7]:

    check_null_values_in_data()

    # ## Data Preprocessing

    # ### Drop rows with nulls in target feature 'Spending on healthy eating'

    # In[8]:

    yps_data.dropna(subset=[var_of_interest], inplace=True)

    # ### Discritization of continuous values

    # In[9]:

    def binning():

        labels = [1.0, 2.0, 3.0, 4.0, 5.0]

        weight_bins = [0, 54, 60, 68, 79, 300]
        yps_data['Weight'] = pd.cut(yps_data['Weight'], bins=weight_bins, labels=labels)
        yps_data['Weight'].fillna(3.0, inplace=True)

        height_bins = [0, 165, 170, 176, 184, 300]
        yps_data['Height'] = pd.cut(yps_data['Height'], bins=height_bins, labels=labels)
        yps_data['Height'].fillna(3.0, inplace=True)

        age_bins = [0, 18, 19, 20, 22, 40]
        yps_data['Age'] = pd.cut(yps_data['Age'], bins=age_bins, labels=labels)
        yps_data['Age'].fillna(3.0, inplace=True)

        number_of_siblings_bins = [-1, 0, 1, 2, 3, 15]
        yps_data['Number of siblings'] = pd.cut(yps_data['Number of siblings'], bins=number_of_siblings_bins,
                                                labels=labels)
        yps_data['Number of siblings'].fillna(3.0, inplace=True)
        # return yps_data

    # In[10]:

    binning()

    # ### Preprocessing of categorical data

    # #### Find unique values of all categorical features

    # In[11]:

    def find_unique_values():

        unique_cat_vals = {'Gender': yps_data['Gender'].unique(),
                           'Left - right handed': yps_data['Left - right handed'].unique(),
                           'Education': yps_data['Education'].unique(),
                           'Only child': yps_data['Only child'].unique(),
                           'Village - town': yps_data['Village - town'].unique(),
                           'House - block of flats': yps_data['House - block of flats'].unique(),
                           'Smoking': yps_data['Smoking'].unique(),
                           'Alcohol': yps_data['Alcohol'].unique(),
                           'Punctuality': yps_data['Punctuality'].unique(),
                           'Lying': yps_data['Lying'].unique(),
                           'Internet usage': yps_data['Internet usage'].unique()}
        return unique_cat_vals

    # In[12]:

    find_unique_values()

    # #### Convert unique text values to numbers for categorical data

    # In[13]:

    def categorical_data_processing():

        yps_data['Gender'] = yps_data['Gender'].map({'male': 1, 'female': 2})
        yps_data['Left - right handed'] = yps_data['Left - right handed'].map({'right handed': 1, 'left handed': 2})
        yps_data['Education'] = yps_data['Education'].map(
            {'currently a primary school pupil': 1.0, 'primary school': 2.0, 'secondary school': 3.0,
             'college/bachelor degree': 4.0, 'masters degree': 5.0, 'doctorate degree': 6.0})
        yps_data['Only child'] = yps_data['Only child'].map({'no': 1, 'yes': 2})
        yps_data['Village - town'] = yps_data['Village - town'].map({'village': 1, 'city': 2})
        yps_data['House - block of flats'] = yps_data['House - block of flats'].map(
            {'block of flats': 1, 'house/bungalow': 2})
        yps_data['Smoking'] = yps_data['Smoking'].map(
            {'never smoked': 1.0, 'tried smoking': 2.0, 'former smoker': 3.0, 'current smoker': 4.0})
        yps_data['Alcohol'] = yps_data['Alcohol'].map({'never': 1.0, 'social drinker': 2.0, 'drink a lot': 3.0})
        yps_data['Punctuality'] = yps_data['Punctuality'].map(
            {'i am often early': 1.0, 'i am always on time': 2.0, 'i am often running late': 3.0})
        yps_data['Lying'] = yps_data['Lying'].map(
            {'never': 1.0, 'only to avoid hurting someone': 2.0, 'sometimes': 3.0, 'everytime it suits me': 4.0})
        yps_data['Internet usage'] = yps_data['Internet usage'].map(
            {'no time at all': 1.0, 'less than an hour a day': 2.0, 'few hours a day': 3.0, 'most of the day': 4.0})

    # In[14]:

    categorical_data_processing()

    # ###  Replace nan with its numpy quivalent and impute all NaN with most frequent value for that particular feature in the data

    # In[15]:

    yps_data = yps_data.replace("nan", np.nan)
    yps_data = yps_data.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(yps_data)
    yps_data_imputed = imp.transform(yps_data)
    yps_data = pd.DataFrame(data=yps_data_imputed[:, :], index=[i for i in range(len(yps_data_imputed))],
                            columns=yps_data.columns.tolist())

    # ## Correlation

    # ### Find correlation of all features with target feature

    # In[16]:

    def plot_correlation(x, y, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Features")
        sns.barplot(x=x, y=y, ax=ax)
        ax.set_ylabel("Correlation coefficients")
        plt.show()

    def find_correlation(var_of_interest, yps_data):

        df = copy.deepcopy(yps_data)

        cols = [col for col in df.columns]
        cols.remove(var_of_interest)
        lbls, vals = [], []
        for col in cols:
            lbls.append(col)
            vals.append(np.corrcoef(df[col], df[var_of_interest])[0, 1])
        correlations = pd.DataFrame({'features': lbls, 'corr_values': vals})
        correlations = correlations.sort_values(by='corr_values')
        print("")
        print("Features and their correlation values with target feature:")
        print(correlations)
        print("")
        return correlations

        # In[17]:

    corrs = find_correlation(var_of_interest, yps_data)

    # In[18]:

    plot_correlation(corrs.corr_values, corrs['features'], figsize=(10, 25))

    # ### The strongest and weakest correlation with target feature

    # In[19]:

    print("Strongest correlation:")
    display(corrs[corrs.corr_values == max(corrs.corr_values)])
    print("Weakest correlation:")
    display(corrs[corrs.corr_values == min(corrs.corr_values)])

    # ### Top ten positive correlated features

    # In[20]:

    print()
    print("Top ten positive correlated features")
    print(corrs.tail(10)['features'])

    # ### Top five negative correlated features

    # In[22]:

    print()
    print("Top five negative correlated features")
    print(corrs.tail(5)['features'])

    # ### Get correlated features for the target feature

    # In[23]:

    def get_correlated_features(corrs):

        top_pos_corr_features = corrs.tail(8)['features']
        top_neg_corr_features = corrs.head(5)['features']
        top_corelated_features = np.concatenate((top_pos_corr_features.values, top_neg_corr_features.values), axis=0)
        # feature_matrix = topCorrelatedfeatures.as_matrix()

        top_corr_features_list = list(top_corelated_features)
        print("")
        print("Selected top correlated features for feature matrix:")
        print(top_corr_features_list)
        return top_corr_features_list

    # In[24]:

    top_corr_features_list = get_correlated_features(corrs)

    # ## Machine learning

    # ### Get feature matrix and target labels to train the models

    # In[25]:

    def get_feature_matrix_and_labels(top_corr_features_list):

        correlated_yps_data = yps_data[top_corr_features_list].copy()
        rest_all_yps_data = yps_data.drop(columns=[var_of_interest])

        feature_matrix = correlated_yps_data.as_matrix()
        print("feature_matrix shape: ", feature_matrix.shape)
        # print(type(feature_matrix))

        labels = list(yps_data[var_of_interest])
        return feature_matrix, labels

    # In[26]:

    feature_matrix, labels = get_feature_matrix_and_labels(top_corr_features_list)

    # ### Split the data for training and testing

    # In[27]:

    def split_data_in_train_test(feature_matrix, labels):

        train_vectors, test_vectors, train_labels, test_labels = train_test_split(feature_matrix, labels,
                                                                                  test_size=0.15, random_state=0)

        print("")
        print("train_vectors.size(): ", len(train_vectors))
        print("test_vectors.size(): ", len(test_vectors))
        print("train_labels.size(): ", len(train_labels))
        print("test_labels.size(): ", len(test_labels))

        return train_vectors, test_vectors, train_labels, test_labels

    # In[28]:

    train_vectors, test_vectors, train_labels, test_labels = split_data_in_train_test(feature_matrix, labels)

    # ### Try few machine learning classifiers on these data

    # #### Naive Bayes Classifier

    # In[29]:

    def gaussian_naive_bayes(train_vectors, test_vectors, train_labels, test_labels):
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()

        classifier_gnb_scores = cross_val_score(gnb, train_vectors, train_labels, cv=8)
        print("")
        print('Gaussian naive bayes cross_val_scores: ', classifier_gnb_scores)
        y_pred = gnb.fit(train_vectors, train_labels)
        y_pred = gnb.predict(test_vectors)

        gnb_accuracy = accuracy_score(test_labels, y_pred)
        print("")
        print('Gaussian naive bayes accuracy: ', gnb_accuracy)
        print()
        print(classification_report(test_labels, y_pred))

    # In[30]:

    gaussian_naive_bayes(train_vectors, test_vectors, train_labels, test_labels)

    # #### Support Vector Machine classifiers

    # #### Also check the cross validation scores

    # In[31]:

    def svc_rbf(train_vectors, test_vectors, train_labels, test_labels):

        print("Training and testing on SVC with rbf kernel")
        classifier_rbf = SVC(kernel='rbf', gamma=0.01)

        classifier_rbf_scores = cross_val_score(classifier_rbf, train_vectors, train_labels, cv=8)
        print("")
        print('SVC RBF kernel cross_val_scores: ', classifier_rbf_scores)

        classifier_rbf.fit(train_vectors, train_labels)
        prediction_rbf = classifier_rbf.predict(test_vectors)
        classifier_rbf_accuracy = accuracy_score(test_labels, prediction_rbf)
        print("")
        print('SVC RBF kernel accuracy: ', classifier_rbf_accuracy)
        print()
        print(classification_report(test_labels, prediction_rbf))

    # In[32]:

    svc_rbf(train_vectors, test_vectors, train_labels, test_labels)

    # In[33]:

    def linear_svm(train_vectors, test_vectors, train_labels, test_labels):

        classifier_liblinear = LinearSVC()

        classifier_liblinear_scores = cross_val_score(classifier_liblinear, train_vectors, train_labels, cv=8)
        print("")
        print('Linear SVC cross_val_scores: ', classifier_liblinear_scores)

        classifier_liblinear.fit(train_vectors, train_labels)
        prediction_liblinear = classifier_liblinear.predict(test_vectors)

        classifier_liblinear_accuracy = accuracy_score(test_labels, prediction_liblinear)
        print("")
        print('Linear SVC accuracy: ', classifier_liblinear_accuracy)
        print()
        print(classification_report(test_labels, prediction_liblinear))

    # In[34]:

    linear_svm(train_vectors, test_vectors, train_labels, test_labels)

    # #### Random Forest Classifier

    # In[35]:

    def rfc(train_vectors, test_vectors, train_labels, test_labels):

        from sklearn.ensemble import RandomForestClassifier as RFR

        rfr = RFR()

        rfr_scores = cross_val_score(rfr, train_vectors, train_labels, cv=10)
        print("")
        print('Random forest classifier crosss_val_scores: ', rfr_scores)

        rfr.fit(train_vectors, train_labels)
        y_pred = rfr.predict(test_vectors)

        rfr_accuracy = accuracy_score(test_labels, y_pred)
        print("")
        print('Random forest classifier accuracy: ', rfr_accuracy)
        print()
        print(classification_report(test_labels, y_pred))

    # In[36]:

    rfc(train_vectors, test_vectors, train_labels, test_labels)

    # ### Accuracy is very low with all the classifiers. We might have to do some more data pre processing.

    # #### Check the data size and count of each unique values in the target labels

    # In[37]:

    print("")
    print("Data size: ", len(labels))
    print("")
    l_vals_count = yps_data[var_of_interest].value_counts(normalize=False, sort=True, ascending=False, bins=None,
                                                          dropna=True)
    print("Count of each unique values in the target labels:")
    print(l_vals_count)

    # #### We can see that the classes are highly imbalanced.
    # Count of examples of class 1.0 is only 41 wheras count of examples of class 4.0 is 330.
    # These imbalanced classes could be a problem for Machine learning algorithms.
    # We can do undersampling or oversampling to fix this issue. Since the data size is small, oversampling can result in better classification.

    # #### Lets do oversampling of the data to balance all the classes by using SMOTE API from imblearn.over_sampling package.

    # In[38]:

    def resample_data(feature_matrix, labels):

        print()
        print("Oversampling the data to balance all the classes")
        from imblearn.over_sampling import SMOTE
        X_resampled, y_resampled = SMOTE().fit_sample(feature_matrix, labels)
        print("")
        print("Data examples classes and their counts:")
        print(sorted(Counter(y_resampled).items()))
        return X_resampled, y_resampled

    # In[39]:

    feature_matrix, labels = resample_data(feature_matrix, labels)

    # ### Split the data again for resampled data for training and testing

    # In[40]:

    train_vectors, test_vectors, train_labels, test_labels = split_data_in_train_test(feature_matrix, labels)

    # ### Now run the classifiers again with resampled train and test data

    # In[45]:

    print("Running gaussian_naive_bayes after oversampling the data")
    gaussian_naive_bayes(train_vectors, test_vectors, train_labels, test_labels)

    # In[46]:

    print("Running svc_rbf after oversampling the data")
    svc_rbf(train_vectors, test_vectors, train_labels, test_labels)

    # In[47]:

    linear_svm(train_vectors, test_vectors, train_labels, test_labels)

    # In[48]:

    rfc(train_vectors, test_vectors, train_labels, test_labels)

    # ### We can see that we have achieved far better accuracy than earlier after oversampling the data

    # ### Now lets try to tune the parameters for better accuracy

    # In[49]:

    def find_best_svc_params():

        # Set the parameters by cross-validation
        tuning_params = [{'kernel': ['rbf'], 'gamma': ['auto', 1e-3, 1e-4],
                          'C': [1, 100, 500]},
                         {'kernel': ['linear'], 'C': [1, 100, 500]}]

        print("Tuning hyper-parameters for SVC")
        print()
        clf = GridSearchCV(SVC(), tuning_params, cv=5)
        clf.fit(train_vectors, train_labels)

        print("Best hyper-parameters found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Cross validation scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        return clf.best_params_

    # In[50]:

    svc_best_params = find_best_svc_params()

    # In[51]:

    def plot_validation_curve(estimator, X, y, title):

        from sklearn.model_selection import validation_curve

        param_range = np.logspace(-6, -1, 5)
        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name="gamma", param_range=param_range,
            cv=10, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title(title)
        plt.xlabel("$\gamma$")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.show()

    # In[66]:

    def svc_best_params_classifier(train_vectors, test_vectors, train_labels, test_labels, best_params):

        if not best_params:
            classifier_svc = SVC(kernel='rbf')
        else:
            kernel = best_params.get('kernel')
            if not best_params.get('gamma'):
                gamma = 'auto'
            else:
                gamma = best_params.get('gamma')
            C = best_params.get('C')
            classifier_svc = SVC(kernel=kernel, gamma=gamma, C=C)

        print()
        print("Best params used in SVC classifier for final classification:")
        print("kernel: ", kernel)
        print("gamma: ", gamma)
        print("C: ", C)

        title = "Validation Curves with (SVM, RBF kernel)"

        plot_validation_curve(classifier_svc, train_vectors, train_labels, title)

        classifier_svc.fit(train_vectors, train_labels)
        y_pred = classifier_svc.predict(test_vectors)

        misclassified = np.where(np.asarray(test_labels) != y_pred)
        print()
        print("Misclassified examples indices in test vector:", misclassified)
        classifier_svc_accuracy = accuracy_score(test_labels, y_pred)

        print("")
        print('SVC RBF kernel accuracy: ', classifier_svc_accuracy)
        print()
        print(classification_report(test_labels, y_pred))

    # In[67]:

    print("Training and testing on SVC with best params found from cross validation")
    svc_best_params_classifier(train_vectors, test_vectors, train_labels, test_labels, svc_best_params)

    # In[56]:

    def find_best_rfc_params():

        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(random_state=0)

        param_grid = {
            'n_estimators': [10, 500, 700]}

        print("Tuning hyper-parameters")
        print()
        clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=8)
        clf.fit(train_vectors, train_labels)

        print("Best hyper-parameters found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Cross validation scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        return clf.best_params_

    # In[57]:

    rfc_best_params = find_best_rfc_params()

    # In[64]:

    def rfc_classifier(train_vectors, test_vectors, train_labels, test_labels, best_params):

        print()
        if not best_params or len(best_params) == 0:
            print("No best params. Using default params.")
            rfr = RFR(n_estimators=500, random_state=0)
        else:
            n_estimators = best_params.get('n_estimators')
            print("Best params found for Random forest classifier - n_estimators: ", n_estimators)
            rfr = RFR(n_estimators=n_estimators, random_state=0)

        rfr.fit(train_vectors, train_labels)

        y_pred = rfr.predict(test_vectors)

        misclassified = np.where(np.asarray(test_labels) != y_pred)
        print()
        print("Misclassified examples indices in test vector:", misclassified)

        rfr_accuracy = accuracy_score(test_labels, y_pred)
        print("")
        print('Random forest classifier accuracy after parameter tuning: ', rfr_accuracy)
        print()
        print(classification_report(test_labels, y_pred))
        print()

    # In[65]:

    print("Training and testing Random forest classifier with best params found from cross validation")
    rfc_classifier(train_vectors, test_vectors, train_labels, test_labels, rfc_best_params)

if __name__ == '__main__':
    spend_money_heathy_food()