from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef
import numpy as np

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
        model, space, scoring="roc_auc", n_iter=1, n_jobs=-1, cv=cv, random_state=1
    )
    result = search.fit(X_train, y_train)
    return result.best_params_


def get_best_features(feature_names, model_fit, X_test, y_test):
    PI = permutation_importance(
        model_fit, X_test, y_test, n_repeats=10, random_state=42
    )
    important_features = []
    for q in PI.importances_mean.argsort()[::-1]:
        if PI.importances_mean[q] - PI.importances_std[q] > 0:
            important_features.append(
                feature_names[q]
            )  # works cos they are in same order as the x columns
    return important_features


oversample = RandomOverSampler(sampling_strategy=0.5, random_state=42)


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
        X_test, y_test = oversample.fit_resample(X_test, y_test)

    return (test_data, train_data, X_train, y_train, X_test, y_test, cipro_R_prev)


def get_feature_effects(feature_names, model_fit, X_test, y_test):
    PI = permutation_importance(
        model_fit, X_test, y_test, n_repeats=10, random_state=42
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
