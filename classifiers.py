import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd


def fn_search_best_svm_classifier(x, y, nfolds, feature_name, display_results=False):

    """ This function searchs the svm model the achieved the highest F1-score value.

    :param x: feature matrix that contains numerical information about the knowledge base.
           y: ndarray with the labels of the intent names.

    :return: pandas dataframe with the best parameters C and gamma for svm model.

    """

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                        ]

    svm_model = GridSearchCV(SVC(), tuned_parameters, cv=nfolds,  scoring='%s_macro' % 'f1')
    svm_model.fit(x, y)

    means = svm_model.cv_results_['mean_test_score']
    stds = svm_model.cv_results_['std_test_score']
    parameters = svm_model.cv_results_['params']

    aux = pd.concat((pd.DataFrame(means), pd.DataFrame(stds)), axis=1)
    aux.columns = [[feature_name, feature_name], ['Mean', 'STD']]

    result = pd.concat((pd.DataFrame.from_dict(parameters), aux), axis=1)

    max_f1 = np.amax(means)
    idx = np.array(np.where(means == max_f1))

    best_svm_parameters = result.iloc[idx[0]]

    best_kernel = best_svm_parameters['kernel'].values
    best_c = best_svm_parameters['C'].values
    best_gamma = best_svm_parameters['gamma'].values

    classifier = SVC(kernel=best_kernel, gamma=best_gamma, C=best_c).fit(x, y)

    if display_results:
        print("Best parameters set found for SVM using " + feature_name + ' as feature: ')
        print()
        print(result.iloc[idx[0]])
        print('------------------------------------------------------------------------')

    # best_svm_parameters.reset_index(drop=True)

    return classifier

