import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.decomposition import PCA

from config.features import CAT_FEATURES


def preprocess_data(X_train, X_test=None, pca_components=90):
    """
    Preprocesses training and optionally testing data by:
    - Replacing infinite values with NaN.
    - Encoding categorical features.
    - Scaling numerical features.
    - Applying PCA to features whose names start with 'V'.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training dataset.
    X_test : pd.DataFrame, optional
        Testing dataset.
    pca_components : int, default=90
        Number of principal components to retain for PCA on features starting with 'V'.

    Returns
    -------
    X_train_transformed : np.ndarray
        The transformed training data.
    X_test_transformed : np.ndarray, optional
        The transformed testing data, only returned if X_test is provided.

    Notes
    -----
    - Categorical features are derived from the global 'CAT_FEATURES' list as well as all object-type columns in X_train.
    - Features starting with 'V' undergo PCA, while other numeric features are scaled but not reduced.
    - After preprocessing, categorical features are encoded and numeric features are scaled.
    """
    pca_features = X_train.columns[X_train.columns.str.startswith('V')]
    cat_features = np.unique(CAT_FEATURES + X_train.select_dtypes(include=['object']).columns.tolist())
    num_features = [col for col in X_train.columns if col not in cat_features and col not in pca_features]

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)

    if X_test is not None:
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    categorical_transformer = Pipeline(steps=[
        ('to_string', FunctionTransformer(lambda X: X.astype(str))),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    pca_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_components))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_features),
            ('num', numeric_transformer, num_features),
            ('pca', pca_transformer, pca_features)
        ],
        remainder='passthrough'
    )

    X_train_transformed = preprocessor.fit_transform(X_train)

    if X_test is not None:
        X_test_transformed = preprocessor.transform(X_test)

        del X_test
        return X_train_transformed, X_test_transformed
    
    del X_train
    return X_train_transformed