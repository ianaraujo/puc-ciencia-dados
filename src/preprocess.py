import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from config.features import CAT_FEATURES


def preprocess_data(X_train, X_test=None, cat_features=None):
    """
    Preprocess training and testing data by handling infinite values, 
    encoding categorical features, and scaling numerical features.

    Parameters:
    - X_train: pd.DataFrame, Training dataset.
    - X_test: pd.DataFrame, Testing dataset (optional).
    - cat_features: list, Names of categorical features.
    - num_features: list, Names of numerical features.

    Returns:
    - X_train_transformed: np.ndarray, Transformed training data.
    - X_test_transformed: np.ndarray, Transformed testing data (if X_test is provided).
    """
    X_train = X_train.replace([np.inf, -np.inf], np.nan)

    if X_test is not None:
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

    cat_features = CAT_FEATURES + [col for col in X_train.select_dtypes(include=['object']).columns if col not in CAT_FEATURES]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_features),
            ('num', numeric_transformer, [col for col in X_train.columns if col not in cat_features])
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)

    if X_test is not None:
        X_test_transformed = preprocessor.transform(X_test)

        return X_train_transformed, X_test_transformed

    return X_train_transformed
