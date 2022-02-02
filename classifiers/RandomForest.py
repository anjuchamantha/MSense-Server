import pandas as pd
from matplotlib import pyplot as plt
from sklearn import ensemble
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix


def rf(x_train, y_train, x_test, y_test):
    print("\n[RF]")
    rf_clf = ensemble.RandomForestClassifier(n_estimators=200)

    rf_clf.fit(x_train, y_train.values.ravel())

    # rf_score = rf_clf.score(x_test, y_test)

    y_pred_test = rf_clf.predict(x_test)
    y_pred_test_prob = rf_clf.predict_proba(x_test)
    # print(classification_report(y_test, y_pred_test))

    accuracy = accuracy_score(y_test, y_pred_test)
    print('Accuracy: %f' % accuracy)

    f1_w = f1_score(y_true=y_test, y_pred=y_pred_test, average='weighted')
    f1_avg = f1_score(y_true=y_test, y_pred=y_pred_test, average='macro')
    print('F1 score w: %f' % f1_w)
    print('F1 score a: %f' % f1_avg)

    auc_0 = roc_auc_score(y_true=y_test, y_score=y_pred_test_prob[:, 0])
    auc_1 = roc_auc_score(y_true=y_test, y_score=y_pred_test_prob[:, 1])
    print('ROC AUC 0: %f' % auc_0)
    print('ROC AUC 1: %f' % auc_1)

    kappa = cohen_kappa_score(y_test, y_pred_test)
    print('Kappa : %f' % kappa)

    matrix = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = matrix.ravel()
    print(matrix)
    print("tp fn fp tn\n", tp, fn, fp, tn)

    return [accuracy, f1_w, f1_avg, auc_1, kappa, tp, fn, fp, tn]


def plot_rf_feature_importance(x_train, y_train, x_test, y_test, x_cols):
    rf_clf = ensemble.RandomForestClassifier(n_estimators=200)
    rf_clf.fit(x_train, y_train.values.ravel())
    rf_score = rf_clf.score(x_test, y_test)
    print("\nScore", rf_score)

    df = pd.DataFrame({"Feature": x_cols,
                       "Importance": rf_clf.feature_importances_})

    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values('Importance', ascending=False)
    print(df_sorted)

    plt.bar('Feature', 'Importance', data=df_sorted)
    plt.xticks(rotation=90)
    # plt.savefig("bar_plot_matplotlib_descending_order_Python.png")
    plt.show()


def get_feature_importance(x_train, y_train, x_test, y_test, x_cols):
    # print("[RF - Feature Importance]")

    rf_clf = ensemble.RandomForestClassifier(n_estimators=50)
    rf_clf.fit(x_train, y_train.values.ravel())
    rf_clf.score(x_test, y_test)

    return pd.DataFrame({"Feature": x_cols,
                         "Importance": rf_clf.feature_importances_})
