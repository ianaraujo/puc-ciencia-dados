import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from config.features import CAT_FEATURES

def remove_inf(X_train, X_test=None):
    """
    Transform 'inf' values into 'nan'.

    Parameters:
    - X_train: pd.DataFrame, Training dataset.
    - X_test: pd.DataFrame, Testing dataset (optional).

    Returns:
    - X_train_transformed: np.ndarray, Transformed training data.
    - X_test_transformed: np.ndarray, Transformed testing data (if X_test is provided).
    """
    X_train_transformed = X_train.replace([np.inf, -np.inf], np.nan)

    if X_test is not None:
        X_test_transformed = X_test.replace([np.inf, -np.inf], np.nan)
        
        return X_train_transformed, X_test_transformed

    return X_train_transformed


def preprocess_data(X_train, X_test=None):
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
    cat_features = [col  for col in CAT_FEATURES if col in X_train.columns] 
    object_features = [col for col in X_train.select_dtypes(include=['object']).columns if col not in CAT_FEATURES]

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
            ('cat', categorical_transformer, cat_features + object_features),
            ('num', numeric_transformer, [col for col in X_train.columns if col not in cat_features + object_features])
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    # ---- Checks and prints ----
    print("Feature names after transformation:")
    print(feature_names)
    print()

    # Print a small sample of transformed data (first 5 rows)
    print("Sample of transformed X_train (first 5 rows):")
    print(X_train_transformed[:5, :])
    print()

    # Print summary statistics for the transformed data
    means = X_train_transformed.mean(axis=0)
    stds = X_train_transformed.std(axis=0)
    mins = X_train_transformed.min(axis=0)
    maxs = X_train_transformed.max(axis=0)

    print("Summary statistics for transformed X_train:")
    print(f"Means: {means}")
    print(f"Stds:  {stds}")
    print(f"Mins:  {mins}")
    print(f"Maxs:  {maxs}")
    print()

    if X_test is not None:
        X_test_transformed = preprocessor.transform(X_test)

        return X_train_transformed, X_test_transformed

    return X_train_transformed
