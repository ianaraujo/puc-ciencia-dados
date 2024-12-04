from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from config.features import CAT_FEATURES

def preprocess_data(X_train, X_test=None):
    """
    Preprocess training and testing data by handling infinite values, 
    encoding categorical features, and imputing missing values.

    Parameters:
    - X_train: pd.DataFrame, Training dataset.
    - X_test: pd.DataFrame, Testing dataset (optional).

    Returns:
    - X_train_transformed: np.ndarray, Transformed training data.
    - X_test_transformed: np.ndarray, Transformed testing data (if X_test is provided).
    """
    X_train[CAT_FEATURES] = X_train[CAT_FEATURES].astype(str)
    
    if X_test is not None:
        X_test[CAT_FEATURES] = X_test[CAT_FEATURES].astype(str)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, [col for col in X_train.columns if col not in CAT_FEATURES]),
            ('cat', categorical_transformer, CAT_FEATURES)
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    
    if X_test is not None:
        X_test_transformed = preprocessor.transform(X_test)
        return X_train_transformed, X_test_transformed

    return X_train_transformed