import time
import pickle
import argparse
import pandas as pd
import numpy as np
from src.preprocess import FeatureTransform, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, fbeta_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler


def read_train(dir: str = 'data/') -> pd.DataFrame:
    merged_chunks = []

    train_identity = pd.read_csv(dir + 'train_identity.csv')
    transaction_chunks = pd.read_csv(dir + 'train_transaction.csv', chunksize=10 ** 5)

    for chunk in transaction_chunks:
        new_chunk = chunk.merge(train_identity, on='TransactionID', how='left')
        merged_chunks.append(new_chunk)

    train = pd.concat(merged_chunks)
    train['hasIdentity'] = (train[train_identity.columns.difference(['TransactionID'])].isna().all(axis=1) == False).astype(int)
    
    return train 


def split_data(dataset: pd.DataFrame):
    X = dataset.drop(columns=['isFraud'])
    y = dataset['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    del dataset

    return X_train, X_test, y_train, y_test


def save_model(model, output_path):
    """Save the trained model using pickle."""
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {output_path}")


def main(args):
    seed = 77
    np.random.seed(seed)

    start_time = time.time()

    train = read_train(dir='data/')
    train = FeatureTransform.create_features(train)

    X_train, X_test, y_train, y_test = split_data(train)
    X_train_transformed, X_test_transformed = preprocess_data(X_train, X_test)
    
    undersampler = RandomUnderSampler(sampling_strategy=0.33, random_state=seed)
    X_train_transformed, y_train = undersampler.fit_resample(X_train_transformed, y_train)

    clf = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=1000, 
        max_depth=15,
        max_features=None,
        min_samples_split=15,
        n_jobs=args.n_jobs, 
        random_state=seed,
    )

    print('Training model...')
    clf.fit(X_train_transformed, y_train)

    y_pred = clf.predict(X_test_transformed)
    y_pred_proba = clf.predict_proba(X_test_transformed)[:, 1]

    print('Evaluating on validation set...')
    print()

    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

    print("## Confusion Matrix (Markdown Table)\n")
    print("|                  | Predicted: Negative | Predicted: Positive |")
    print("|------------------|---------------------|----------------------|")
    print(f"| **Actual: Negative** | {TN}                   | {FP}                    |")
    print(f"| **Actual: Positive** | {FN}                   | {TP}                    |")
    print()

    auc_score = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    print('ROC AUC:   ', auc_score)
    print('Precision: ', precision)
    print('Recall:    ', recall)
    print('F2 Score:  ', f2)
    print()

    if args.output:
        save_model(clf, output_path=args.output)

    total_time = time.time() - start_time
    print(f"Model training completed in {total_time:.2f} seconds")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RandomForest Classification Model")
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs to run for model training')
    parser.add_argument('--output', type=str, default=None, help='Path to save the trained model')
    
    args = parser.parse_args()

    main(args)