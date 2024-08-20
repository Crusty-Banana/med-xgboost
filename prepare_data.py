import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb

def prepare_data_for_xgboost(df):
    """
    Transforms a DataFrame with patient data into a format suitable for XGBoost.
    
    Parameters:
    - df: DataFrame containing patient data with columns: 'AGE', 'SEX', 'PATHOLOGY', 'EVIDENCES', 
          'INITIAL_EVIDENCE', 'DIFFERENTIAL_DIAGNOSIS'.
    
    Returns:
    - dtrain: DMatrix object for XGBoost.
    - pathology_encoder: LabelEncoder for PATHOLOGY, useful for decoding predictions.
    """
    
    # Encode SEX
    df['SEX'] = df['SEX'].map({'Male': 0, 'Female': 1})

    # Encode PATHOLOGY
    pathology_encoder = LabelEncoder()
    df['PATHOLOGY'] = pathology_encoder.fit_transform(df['PATHOLOGY'])

    # Encode EVIDENCES using one-hot encoding
    evidence_list = df['EVIDENCES'].apply(lambda x: x.str.strip('[]').replace("'", "").str.split(', '))
    print(evidence_list)

    # # One-hot encode the EVIDENCES
    # evidence_df = pd.get_dummies(evidence_list.apply(pd.Series).stack()).sum(level=0)
    # # Concatenate the original DataFrame with the one-hot encoded evidences
    # df = pd.concat([df.drop(['EVIDENCES'], axis=1), evidence_df], axis=1)

    # # Handle DIFFERENTIAL_DIAGNOSIS (example: take the pathology with the highest probability)
    # df['TOP_PATHOLOGY'] = df['DIFFERENTIAL_DIAGNOSIS'].apply(lambda x: max(x, key=lambda y: y[1])[0])
    # df['TOP_PATHOLOGY'] = pathology_encoder.transform(df['TOP_PATHOLOGY'])

    # # Drop unnecessary columns like INITIAL_EVIDENCE and DIFFERENTIAL_DIAGNOSIS
    # df = df.drop(['INITIAL_EVIDENCE', 'DIFFERENTIAL_DIAGNOSIS'], axis=1)

    # # Separate features and labels
    # X = df.drop('PATHOLOGY', axis=1)  # Features
    # y = df['PATHOLOGY']  # Labels

    # # Convert to DMatrix for XGBoost
    # dtrain = xgb.DMatrix(X, label=y)

    # return dtrain, pathology_encoder

# Example usage:
# df = pd.read_json('your_data_file.json')  # Load your data into a DataFrame
# dtrain, pathology_encoder = prepare_data_for_xgboost(df)
# Now you can use dtrain in XGBoost
