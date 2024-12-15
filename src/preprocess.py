import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.decomposition import PCA

from config.features import CAT_FEATURES

class FeatureTransform:
    amount = 'TransactionAmt'
    log_amount = 'TransactionAmtLog'
    selected_cols = ['card1', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'addr1']
    start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')

    @classmethod
    def create_features(cls, data: pd.DataFrame) -> pd.DataFrame:
        data.loc[~data['card6'].isin(['credit', 'debit']), 'card6'] = np.nan

        data['TransactionAmtCents'] = (data[cls.amount] - np.floor(data[cls.amount])).astype(np.float64).round(2)
        data[cls.log_amount] = np.log(data[cls.amount])

        for col in cls.selected_cols:
            data[f'amount_mean_{col}'] = data[cls.amount] / data.groupby(col)[cls.amount].transform('mean')
            data[f'amount_std_{col}'] = data[cls.amount] / data.groupby(col)[cls.amount].transform('std')

        for col in cls.selected_cols:
            data[f'log_amount_mean_{col}'] = data[cls.log_amount] / data.groupby(col)[cls.log_amount].transform('mean')
            data[f'log_amount_std_{col}'] = data[cls.log_amount] / data.groupby(col)[cls.log_amount].transform('std')

        data['Day'] = data['TransactionDT'].apply(lambda dt: cls.start_date + timedelta(seconds=dt)).dt.day
        data['Weekday'] = data['TransactionDT'].apply(lambda dt: cls.start_date + timedelta(seconds=dt)).dt.dayofweek
        data['Hour'] = data['TransactionDT'].apply(lambda dt: cls.start_date + timedelta(seconds=dt)).dt.hour

        return data


def preprocess_data(X_train, X_test=None, pca_components=90):
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