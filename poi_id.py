#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import matplotlib.pyplot
from pprint import pprint

# This subroutine calculates value_x / (value_x+value_y), and handles NaN appropriatly.
def ratio (value_x, value_y):
    if (value_x == "NaN"):
        return 0
    elif (value_y == "NaN"):
        return 1
    else:
        return ((value_x+0.0)/(value_x + value_y))

# This subroutine calculates num / denom in a NaN-safe way.
def ratio2 (num, denom):
    if (num == "NaN"):
        return 0
    else:
        return ((num+0.0)/(denom))

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# New features:
# 'proportion_of_stock': Proportion of stock in that person's payment package, i.e. total_stock_value/(total_stock_value + total_payments)
# 'proportion_from_this_to_poi': from_this_person_to_poi / from_messages
# 'proportion_from_poi_to_this': from_poi_to_this_person / to_messages
# 'proportion_payments_deferred': deferred_income / (total_payments + deferred_income)
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                          'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                          'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                          'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Based on the Outlier section, a TOTAL datapoint was mixed into this set. This needs to be removed.
del my_dataset['TOTAL']

sorted_keys = numpy.sort (my_dataset.keys())
data = featureFormat(my_dataset, features_list, sort_keys = True)
for i in range (len(data)):
    if (
        (data[i][features_list.index('restricted_stock')] < abs(data[i][features_list.index('restricted_stock_deferred')])) |
        (data[i][features_list.index('exercised_stock_options')] > data[i][features_list.index('total_stock_value')])
        ):
        print ("Stock value inconsistency detected for:", sorted_keys[i])

# # ('Stock value inconsistency detected for:', 'BELFER ROBERT')
# # ('Stock value inconsistency detected for:', 'BHATNAGAR SANJAY')

# Fixing Sanjay Bhatnagar and Robert Belfer's finalcials based on the PDF data.
my_dataset['BHATNAGAR SANJAY']['other'] = 'NaN'
my_dataset['BHATNAGAR SANJAY']['expenses'] = 137864
my_dataset['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
my_dataset['BHATNAGAR SANJAY']['total_payments'] = 137864
my_dataset['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
my_dataset['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
my_dataset['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
my_dataset['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

my_dataset['BELFER ROBERT']['deferred_income']=-102500
my_dataset['BELFER ROBERT']['deferral_payments']= 'NaN'
my_dataset['BELFER ROBERT']['expenses'] = 3285
my_dataset['BELFER ROBERT']['director_fees'] = 102500
my_dataset['BELFER ROBERT']['total_payments'] = 3285
my_dataset['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
my_dataset['BELFER ROBERT']['restricted_stock'] = 44093
my_dataset['BELFER ROBERT']['restricted_stock_deferred'] = -44093
my_dataset['BELFER ROBERT']['total_stock_value'] = 0


nonzero_pct_dict = {}
for i in features_list:
    nonzero_pct_dict [i] =0.0

for one_key in my_dataset.keys():
    my_dataset[one_key]['proportion_of_stock'] = ratio (my_dataset[one_key]['total_stock_value'], my_dataset[one_key]['total_payments'])
    my_dataset[one_key]['proportion_from_this_to_poi'] = ratio2 (my_dataset[one_key]['from_this_person_to_poi'], my_dataset[one_key]['from_messages'])
    my_dataset[one_key]['proportion_from_poi_to_this'] = ratio2 (my_dataset[one_key]['from_poi_to_this_person'], my_dataset[one_key]['to_messages'])
    my_dataset[one_key]['proportion_shared_receipt_with_poi'] = ratio2 (my_dataset[one_key]['shared_receipt_with_poi'], my_dataset[one_key]['to_messages'])
    pos_deferred_income = 'NaN'
    if (my_dataset[one_key]['deferred_income'] != "NaN"):
    # In the data set "deferred_income" always comes in negatives, so it should be converted to positive first before any calculations.
        pos_deferred_income = my_dataset[one_key]['deferred_income'] * -1.0
    my_dataset[one_key]['proportion_payments_deferred'] = ratio (pos_deferred_income, my_dataset[one_key]['total_payments'])
    for one_feature in features_list:
        if my_dataset [one_key] [one_feature] != 'NaN':
            nonzero_pct_dict[one_feature] = (nonzero_pct_dict[one_feature] + 100.0/len(my_dataset))

pprint (nonzero_pct_dict)

# # >>> pprint (nonzero_pct_dict)
# # {'bonus': 56.16438356164386,
# #  'deferral_payments': 26.027397260273947,
# #  'deferred_income': 34.24657534246572,
# #  'director_fees': 10.95890410958904,
# #  'exercised_stock_options': 69.17808219178083,
# #  'expenses': 66.43835616438359,
# #  'from_messages': 58.90410958904113,
# #  'from_poi_to_this_person': 58.90410958904113,
# #  'from_this_person_to_poi': 58.90410958904113,
# #  'loan_advances': 2.73972602739726,
# #  'long_term_incentive': 45.20547945205479,
# #  'other': 63.01369863013703,
# #  'poi': 99.99999999999977,
# #  'restricted_stock': 76.02739726027393,
# #  'restricted_stock_deferred': 12.32876712328767,
# #  'salary': 65.06849315068497,
# #  'shared_receipt_with_poi': 58.90410958904113,
# #  'to_messages': 58.90410958904113,
# #  'total_payments': 85.61643835616427,
# #  'total_stock_value': 86.98630136986289}

### Extract features and labels from dataset for local testing

# Because proportions of POI to/from emails of any insider has been transformed into porportional parameters,
# there's no need to consider 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'
# and 'to_messages' in the downstream analysis.

# features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#                  'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
#                  'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
#                  'proportion_of_stock', 'proportion_from_this_to_poi', 'proportion_from_poi_to_this', 'proportion_payments_deferred',
#                  'proportion_shared_receipt_with_poi'] 
# 
# 
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)

# from sklearn.linear_model import RandomizedLasso
# lasso = RandomizedLasso (random_state = 42, n_jobs=-1)
# lasso.fit(features, labels)

# RandomizedLasso, in the default setting, performs 200 75% stratified subsets of the data, ran Lasso on each,
# and counts the frequency a feature has a non-zero coef_. RandomizedLasse.get_support() gets the indices
# of features having more than 25% probabilities of having non-zero coef_.

# Since a subset of features were selected using the RandomizedLasso method above, the following is performed
# to ensure a copy features and features_list containing only the selected features (and in the latter case, poi).
# numpy is imported as features_list is a numpy.ndarray.

# selected_features_index=lasso.get_support(features)+1
# selected_features_index=numpy.insert (selected_features_index,0,0)
# features_selected=lasso.transform(features)
# selected_features_list=list(numpy.array(features_list)[selected_features_index])

## Net effect of RandomizedLasso feature selection (From line 135 to here):
features_list=['poi', 'salary', 'deferral_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
               'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'proportion_from_this_to_poi']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# Given the Enron dataset has an unbalanced response class, train_test_split needs to have its stratified parameter turned on.
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split (
    features, labels, test_size=0.4, stratify = labels, random_state=42
)


# from sklearn.grid_search import GridSearchCV


# dt_param={'criterion': ['entropy','gini'],
#     'splitter': ['best', 'random'],
#     'max_features':[None, 2,3,4,5,6,7,8,9,10,'auto'],
#     'max_depth':[None, 1,2,3,4,5,6,7,8,9,10],
#     'min_samples_split': [2,3,4,5],
#     'class_weight': [None, 'balanced'],
#     'random_state': [42]}
# dt_clf=DecisionTreeClassifier()
# dt_clf2=GridSearchCV(dt_clf, dt_param, "f1", n_jobs=3, verbose = 20)
# dt_clf2.fit (features_train, labels_train)
# dt_clf2_predict = dt_clf2.predict (features_test)
# dt_clf2.best_score_
# dt_clf2.best_estimator_

# # >>> dt_clf2.best_score_
# # 0.55630252100840338
# # >>> dt_clf2.best_estimator_
# # DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
# #             max_depth=None, max_features=7, max_leaf_nodes=None,
# #             min_samples_leaf=1, min_samples_split=2,
# #             min_weight_fraction_leaf=0.0, presort=False, random_state=42,
# #             splitter='best')


# SVM algorithms require the features to be normalized, thus the use of Pipeline.

# SVC_pipeline = Pipeline([
#     ('norm', Normalizer()),
#     ('clf', SVC()),
# ])
# 
# svc_param={'clf__C': [1.0,10.0,25.0,50.0,100.0],
#     'clf__kernel':['linear', 'poly', 'rbf', 'sigmoid'],
#     'clf__gamma': [1e-10, 1e-5, 1e-2, 'auto'],
#     'clf__probability':[False, True],
#     'clf__class_weight': [None, 'balanced'],
#     'clf__decision_function_shape':['ovo', 'ovr'],
#     'clf__verbose': [True],
#     'clf__random_state': [42]}
# svc_clf2=GridSearchCV(SVC_pipeline, svc_param, "f1", n_jobs=4, verbose=True)
# svc_clf2.fit (features_train, labels_train)
# svc_clf2.best_score_
# svc_clf2.best_estimator_

# # >>> svc_clf2.best_score_
# # 0.2389521373766898
# # >>> svc_clf2.best_estimator_
# # Pipeline(steps=[('norm', Normalizer(copy=True, norm='l2')), ('clf', SVC(C=10.0, cache_size=200, 
# # class_weight='balanced', coef0=0.0,
# #   decision_function_shape='ovo', degree=3, gamma=0.01, kernel='rbf',
# #   max_iter=-1, probability=False, random_state=42, shrinking=True,
# #   tol=0.001, verbose=True))])

# rf_param={'n_estimators': [2,3,4,5,10,25,50,100],
#     'criterion':['gini','entropy'],
#     'max_features':[None, 2,3,4,5,6,7,8,9,10,'auto'],
#     'max_depth':[None, 1,2,3,4,5,6,7,8,9,10],
#     'min_samples_split': [2,3,4,5],
#     'oob_score': [True, False],
#     'class_weight': [None, 'balanced', 'balanced_subsample'],
#     'random_state': [42],
#     }
# rf_clf=RandomForestClassifier()
# rf_clf2=GridSearchCV(rf_clf, rf_param, "f1", verbose=20, n_jobs=4)
# rf_clf2.fit (features_train, labels_train)
# rf_clf2.best_score_
# rf_clf2.best_estimator_

# # >>> rf_clf2.best_score_
# # 0.58554621848739496
# # >>> rf_clf2.best_estimator_
# # RandomForestClassifier(bootstrap=True, class_weight='balanced',
# #             criterion='entropy', max_depth=3, max_features=3,
# #             max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
# #             min_weight_fraction_leaf=0.0, n_estimators=2, n_jobs=1,
# #             oob_score=True, random_state=42, verbose=0, warm_start=False)

# ab_param={'n_estimators': [1,5,10,25,50,100],
#     'learning_rate':[0.25,0.5,0.75,1.0],
#     'random_state': [42]}
# ab_clf=AdaBoostClassifier()
# ab_clf2=GridSearchCV(ab_clf, ab_param, "f1", n_jobs=3, verbose=20)
# ab_clf2.fit (features_train, labels_train)
# ab_clf2.best_score_
# ab_clf2.best_estimator_

# # >>> ab_clf2.best_score_
# # 0.48496732026143791
# # >>> ab_clf2.best_estimator_
# # AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
# #           learning_rate=0.75, n_estimators=25, random_state=42)

# gb_pipeline = Pipeline([
#     ('norm', Normalizer()),
#     ('clf', GaussianNB()),
# ])
# gb_clf=gb_pipeline.fit(features_train, labels_train)
# gb_clf_predict=gb_pipeline.predict(features_test)
# from sklearn.metrics import f1_score
# f1_score(labels_test, gb_clf_predict)

# # >>> f1_score(labels_test, gb_clf_predict)
# # 0.1951219512195122

# Random Forest is selected as the best algorithm; dump the best estimator code here and validate
# The following lines of code below recreates the best estimator identified by GridSearchCV above,
# trains the model, outputs the feature importances, and runs test_classifier on the
# classifier.
clf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='entropy', max_depth=3, max_features=3,
            max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=2, n_jobs=1,
            oob_score=True, random_state=42, verbose=0, warm_start=False)

clf.fit (features_train, labels_train)

print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), features_list[1:]), reverse=True)

# # >>> print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), selected_features_list[1:]), reverse=True)
# # [(0.4581, 'other'), (0.2387, 'expenses'), (0.1432, 'total_stock_value'), (0.0983, 'deferred_income')
# # , (0.0617, 'proportion_from_this_to_poi'), (0.0, 'salary'), (0.0, 'restricted_stock_deferred'), (0.0
# # , 'long_term_incentive'), (0.0, 'exercised_stock_options'), (0.0, 'deferral_payments'), (0.0, 'bonus
# # ')]
test_classifier (clf, my_dataset, features_list)
# >>> test_classifier (clf, my_dataset, selected_features_list)
        # Accuracy: 0.79520       Precision: 0.31965      Recall: 0.47500 F1: 0.38214     F2: 0.43292
        # Total predictions: 15000        True positives:  950    False positives: 2022   False negatives: 1050       True negatives: 10978

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)