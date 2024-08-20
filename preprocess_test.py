# Import libraries
import pandas as pd
import json
import ast
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

train_patients_path = 'data/DDXPlus/release_train_patients.csv'
test_patients_path = 'data/DDXPlus/release_test_patients.csv'
validate_patients_path = 'data/DDXPlus/release_validate_patients.csv'

condition_info_path = 'data/DDXPlus/release_conditions.json'
evidence_info_path = 'data/DDXPlus/release_evidences.json'

train_processed_sample = 'data/processed/train_processed_sample.csv'
train_processed_target = 'data/processed/train_processed_target.csv'

test_processed_sample = 'data/processed/test_processed_sample.csv'
test_processed_target = 'data/processed/test_processed_target.csv'

validate_processed_sample = 'data/processed/validate_processed_sample.csv'
validate_processed_target = 'data/processed/validate_processed_target.csv'

test_df = pd.read_csv(test_patients_path)

def preprocess_df(df, sample_output, target_output):
    # Encode SEX
    df['SEX'] = df['SEX'].map({'M': 0, 'F': 1})

    # Transform the DIFFERENTIAL_DIAGNOSIS into a multi-output format
    # Extract all possible pathologies from the DIFFERENTIAL_DIAGNOSIS column
    df['DIFFERENTIAL_DIAGNOSIS'] = df['DIFFERENTIAL_DIAGNOSIS'].apply(ast.literal_eval)
    all_pathologies = list(set(patho for diag in df['DIFFERENTIAL_DIAGNOSIS'] for patho, _ in diag))

    # Create a DataFrame for the target with one column per pathology, initialized to 0
    target_df = pd.DataFrame(0.0, index=df.index, columns=all_pathologies)

    # Populate the target DataFrame with the probabilities from DIFFERENTIAL_DIAGNOSIS
    for idx, diag in enumerate(df['DIFFERENTIAL_DIAGNOSIS']):
        for patho, proba in diag:
            target_df.at[idx, patho] = proba

    # Drop the original DIFFERENTIAL_DIAGNOSIS column as it's now encoded in target_df
    df = df.drop(['DIFFERENTIAL_DIAGNOSIS'], axis=1)

    # Turn string into list
    df['EVIDENCES_LIST'] = df['EVIDENCES'].str.strip('[]').replace("'", "").str.split(', ')

    # Get all unique evidence in table
    all_evidences = list(set(evidence.replace("'", "") for sublist in df['EVIDENCES_LIST'] for evidence in sublist))

    # Create a DataFrame with all evidence columns initialized to 0
    evidence_df = pd.DataFrame(0, index=df.index, columns=all_evidences)

    # Populate the DataFrame by setting appropriate evidence to 1 where they exist in the row's list
    for evidence in all_evidences:
        evidence_df[evidence] = df['EVIDENCES_LIST'].apply(lambda x: 1 if evidence in x else 0)

    # Concatenate the evidence DataFrame with the original DataFrame
    df = pd.concat([df.drop(['EVIDENCES_LIST'], axis=1), evidence_df], axis=1)

    # Split the data into features (X) and targets (y)
    X = df.drop(columns=['PATHOLOGY', 'EVIDENCES', 'INITIAL_EVIDENCE'])  # Drop PATHOLOGY as it's the ground truth label, not a feature
    y = target_df

    X.to_csv(sample_output, index=False)
    y.to_csv(target_output, index=False)

preprocess_df(test_df, test_processed_sample, test_processed_target)