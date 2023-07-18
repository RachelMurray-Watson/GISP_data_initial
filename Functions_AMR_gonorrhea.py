from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

### Create functions for the gonorrhea AMR paper


def effective_unnecessary_threshold(
    threshold_seq, y_predict_proba, y_test, cipro_R_prevalence
):

    get_effective_threshold = []
    incorrectly_get_X_threshold = []  # no bootstrapping, no 95% CI
    sensitivity_threshold = []
    specificity_threshold = []
    for threshold in threshold_seq:

        y_predict_test = np.where(y_predict_proba[:, 1] > threshold, 1, 0)

        tn_test, fp_test, fn_test, tp_test = confusion_matrix(
            y_true=y_test, y_pred=y_predict_test
        ).ravel()

        sensitivity_test = tp_test / (tp_test + fn_test)
        specificity_test = tn_test / (tn_test + fp_test)

        sensitivity_threshold.append(sensitivity_test * 100)
        specificity_threshold.append(specificity_test * 100)
        get_effective_threshold.append(
            sensitivity_test * cipro_R_prevalence * 100
            + (100 - cipro_R_prevalence * 100)
        )  # q_p
        incorrectly_get_X_threshold.append(
            (100 - cipro_R_prevalence * 100) * (1 - specificity_test)
        )  # c_p"
    return (
        sensitivity_threshold,
        specificity_threshold,
        get_effective_threshold,
        incorrectly_get_X_threshold,
    )


def get_best_hyperparameters(model, cv, space, X_train, y_train):
    search = RandomizedSearchCV(
        model, space, scoring="roc_auc", n_iter=100, n_jobs=-1, cv=cv, random_state=1
    )
    result = search.fit(X_train, y_train)
    return result.best_params_


def get_best_features(feature_names, model_fit, X_test, y_test):
    PI = permutation_importance(
        model_fit, X_test, y_test, n_repeats=100, random_state=42
    )
    important_features = []
    for q in PI.importances_mean.argsort()[::-1]:
        if PI.importances_mean[q] - PI.importances_std[q] > 0:
            important_features.append(
                feature_names[q]
            )  # works cos they are in same order as the x columns
    return important_features


oversample = RandomOverSampler(sampling_strategy="minority", random_state=10)


def get_test_train_data(CIP_data_no_drop, year, feature_names, years_train, model_type):

    train_data = CIP_data_no_drop.loc[CIP_data_no_drop["YEAR"].isin(years_train)]
    X_train = train_data[
        feature_names
    ]  # need to consider all columns BEFORE feature engineering
    y_train = 1 - train_data["Susceptible"]
    # test
    test_data = CIP_data_no_drop.loc[CIP_data_no_drop["YEAR"].isin([year])]
    X_test = test_data[feature_names]
    y_test = 1 - test_data["Susceptible"]
    cipro_R_prev = y_test.sum() / len(y_test)
    if (model_type == 1) | (model_type == 2):
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        # X_test, y_test = oversample.fit_resample(X_test, y_test)
        print("Oversample")
    return (test_data, train_data, X_train, y_train, X_test, y_test, cipro_R_prev)


def get_feature_effects(feature_names, model_fit, X_test, y_test):
    PI = permutation_importance(
        model_fit, X_test, y_test, n_repeats=100, random_state=42
    )

    return PI.importances_mean


def f1_mcc_score_threshold(threshold_seq, y_predict_proba, y_test):

    f1_score_seq = []
    mcc_score_seq = []
    for threshold in threshold_seq:

        y_predict = np.where(y_predict_proba[:, 1] > threshold, 1, 0)

        f1_score_seq.append(f1_score(y_test, y_predict))
        mcc_score_seq.append(matthews_corrcoef(y_test, y_predict))
    return (f1_score_seq, mcc_score_seq)


def encoder_for_GISP(data, column):
    encoder = OneHotEncoder()
    encoder_categories = encoder.fit(data[[column]]).categories_
    encoder_categories = encoder_categories[0].tolist()
    encoder_df = pd.DataFrame(encoder.fit_transform(data[[column]]).toarray())
    combined_data = data.join(encoder_df)
    col_names = list(data.columns) + encoder_categories[0:]
    combined_data.columns = col_names
    return combined_data


#### now try bootstrapping w/ feature selection
iterations = 100
## DO NOT SAMPLE THE TARGET DATA
def bootstrap_auROC(iterations, model, train_data, test_data, y_test, ROC_actual):
    # 1. Find apparent model performance
    bootstrapped_stats = []
    for i in range(iterations):
        # 2. (A) Sample all individuals from training data w/replacement

        sample_train = train_data.sample(
            frac=1, replace=True
        )  ##(a) sample n individuals with replacement

        X_sample_train = sample_train[feature_names]
        y_sample_train = 1 - sample_train["Susceptible"]

        if model_type in [1, 2]:
            X_sample_train, y_sample_train = oversample.fit_resample(
                X_sample_train, y_sample_train
            )

        #  (B) Develop predictive model and find apparent performance of new sample data
        # best_hyperparameters1 = get_best_hyperparameters(model_nn, cv, space, X_train, y_train)
        # model_nn = MLPClassifier(solver = best_hyperparameters1['solver'], activation = best_hyperparameters1['activation'], max_iter = 5000 ,hidden_layer_sizes= best_hyperparameters1['hidden_layer_sizes'], alpha =  best_hyperparameters1['alpha'], random_state=337, learning_rate =best_hyperparameters1['learning_rate'])
        # need original test data - without any feature selection
        X_test_bootstrap = test_data[feature_names]
        model_fit = model.fit(X_sample_train, y_sample_train)
        important_features_sample = get_best_features(
            feature_names, model_fit, X_test_bootstrap, y_test
        )
        while len(important_features_sample) == 0:
            X_sample_train = sample_train[feature_names]
            y_sample_train = 1 - sample_train["Susceptible"]
            if model_type in [1, 2]:
                X_sample_train, y_sample_train = oversample.fit_resample(
                    X_sample_train, y_sample_train
                )
            model_fit = model.fit(X_sample_train, y_sample_train)
            important_features_sample = get_best_features(
                feature_names, model_fit, X_test_bootstrap, y_test
            )
        #
        X_sample_train = X_sample_train[important_features_sample]
        # best_hyperparameters2 = get_best_hyperparameters(model_nn, cv, space, X_sample_train, X_sample_test)
        # model_nn = MLPClassifier(solver = best_hyperparameters2['solver'], activation = best_hyperparameters2['activation'], max_iter = 5000 ,hidden_layer_sizes= best_hyperparameters2['hidden_layer_sizes'], alpha =  best_hyperparameters2['alpha'], random_state=337, learning_rate =best_hyperparameters1['learning_rate'])

        model_fit = model.fit(X_sample_train, y_sample_train)
        model_name = (
            "CIP_bootstrap_" + str(model_type) + "_" + str(year) + "_" + str(i) + ".sav"
        )
        X_data_name = (
            "CIP_bootstrap_X_"
            + str(model_type)
            + "_"
            + str(year)
            + "_"
            + str(i)
            + ".csv"
        )
        y_data_name = (
            "CIP_bootstrap_y_"
            + str(model_type)
            + "_"
            + str(year)
            + "_"
            + str(i)
            + ".csv"
        )
        X_sample_train.to_csv(X_data_name)
        y_sample_train.to_csv(y_data_name)
        pickle.dump(model_fit, open(model_name, "wb"))
        #  (C) Performance of predictive model on original sample (i.e. original training population, X_test, with new selected features)
        X_test_bootstrap = X_test_bootstrap[important_features_sample]
        y_bootstrap_predict = model_fit.predict(X_test_bootstrap)
        ROC_AUC_bootstrap_test_performance = metrics.roc_auc_score(
            y_test, y_bootstrap_predict
        )
        ### (D) Calculate estimate fo variance  by getting (B) - (D)

        difference = (
            ROC_AUC_bootstrap_test_performance - ROC_actual
        )  ## according to https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/resources/mit18_05s14_reading24/

        bootstrapped_stats.append({"Difference": difference})  # ,

    bootstrapped_stats = pd.DataFrame(bootstrapped_stats)
    ## Step 3: Get average optimization

    lower_quartile = np.percentile(bootstrapped_stats["Difference"], 2.5)
    upper_quartile = np.percentile(bootstrapped_stats["Difference"], 97.5)
    ## Step 4: Get optimization-corrected performance

    return lower_quartile, upper_quartile
