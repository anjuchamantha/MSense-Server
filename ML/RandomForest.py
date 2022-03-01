import pickle
from sklearn import ensemble
from sklearn.metrics import accuracy_score, f1_score


def rf_train_and_save(x_train, y_train, x_test, y_test, filename="rf_clf"):
    print("\n[RF TRAIN]")
    rf_clf = ensemble.RandomForestClassifier(n_estimators=200)
    rf_clf.fit(x_train, y_train.values.ravel())

    y_pred_test = rf_clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred_test)
    # print('Accuracy: %f' % accuracy)

    f1_w = f1_score(y_true=y_test, y_pred=y_pred_test, average='weighted')
    # print('F1 score w: %f' % f1_w)

    pickle_out = open(filename + ".pkl", "wb")
    pickle.dump(rf_clf, pickle_out)
    pickle_out.close()
    print("RF Model Saved as: " + filename + ".pkl")

    return_obj = {
        "model": "Random Forest",
        "model_filename": filename + ".pkl",
        "features": "",
        "results": {"accuracy": accuracy, "f1_score": f1_w}
    }

    return return_obj


def rf_predict(predict_data, filename="rf_clf"):
    pickle_in = open(filename + ".pkl", "rb")
    clf = pickle.load(pickle_in)
    prediction = clf.predict(predict_data)

    return_obj = {
        "model": "Random Forest",
        "model_filename": filename + ".pkl",
        "prediction_data": predict_data,
        "prediction": prediction
    }
    return return_obj
