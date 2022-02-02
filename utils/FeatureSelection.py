from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression


def VarianceThresholdFeatureSelection(df):
    print("\nVariance Threshold Feature Selection")
    selector = VarianceThreshold(1)
    selector.fit(df)
    features = df.columns[selector.get_support()]
    print(features)


def SelectKBestFeatureSelection(df, x_cols, y_cols):
    print("\nSelectKBest Feature Selection")
    x = df[x_cols]
    y = df[y_cols]
    # Select top 2 features based on mutual info regression
    selector = SelectKBest(mutual_info_regression, k='all')
    selector.fit(x, y.values.ravel())
    features = x.columns[selector.get_support()]
    print(features)


def RecursiveFeatureElimination(df, x_cols, y_cols):
    print("\nRecursive Feature Elimination")
    x = df[x_cols]
    y = df[y_cols]

    rfe_selector = RFE(estimator=LogisticRegression(solver='lbfgs', max_iter=2000), n_features_to_select=3, step=1)
    rfe_selector.fit(x, y.values.ravel())
    features = x.columns[rfe_selector.get_support()]
    print(features)


def SelectFromModelFeatureSelection(df, x_cols, y_cols):
    print("\nSelectFromModel Feature Selection")
    x = df[x_cols]
    y = df[y_cols]

    sfm_selector = SelectFromModel(estimator=LogisticRegression(solver='lbfgs', max_iter=2000))
    sfm_selector.fit(x, y.values.ravel())
    features = x.columns[sfm_selector.get_support()]
    print(features)


def SequentialBackwardFeatureSelection(df, x_cols, y_cols):
    print("\nSequential Backward Feature Selection")
    x = df[x_cols]
    y = df[y_cols]

    sfs_selector = SequentialFeatureSelector(estimator=LogisticRegression(solver='lbfgs', max_iter=2000),
                                             n_features_to_select=3,
                                             cv=10,
                                             direction='backward')
    sfs_selector.fit(x, y.values.ravel())
    features = x.columns[sfs_selector.get_support()]
    print(features)


# best method
def SequentialForwardFeatureSelection(df, x_cols, y_cols, n_features=3):
    print("\nSequential Forward Feature Selection")
    x = df[x_cols]
    y = df[y_cols]

    sfs_selector = SequentialFeatureSelector(estimator=RandomForestClassifier(n_estimators=50),
                                             n_features_to_select=n_features,
                                             cv=5,
                                             direction='forward', n_jobs=-1)
    sfs_selector.fit(x, y.values.ravel())
    features = x.columns[sfs_selector.get_support()]
    # print(features)
    print(features.tolist())
    return features.tolist()
