import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.special import rel_entr
import numpy as np

train_processed_sample = 'data/processed/train_processed_sample.csv'
train_processed_target = 'data/processed/train_processed_target.csv'

test_processed_sample = 'data/processed/test_processed_sample.csv'
test_processed_target = 'data/processed/test_processed_target.csv'
test_processed_diff = 'data/processed/test_processed_differential_diagnosis.csv'

validate_processed_sample = 'data/processed/validate_processed_sample.csv'
validate_processed_target = 'data/processed/validate_processed_target.csv'

X_train = pd.read_csv(train_processed_sample)
y_train = pd.read_csv(train_processed_target)

X_test = pd.read_csv(test_processed_sample)
y_test = pd.read_csv(test_processed_target)

X_val = pd.read_csv(validate_processed_sample)
y_val = pd.read_csv(validate_processed_target)

# Create DMatrix for model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softprob',  # Multi-class classification with probabilities
    'num_class': 49,  # Number of classes
    'eval_metric': 'mlogloss',  # Multi-class log loss
    'max_depth': 15,  # Maximum depth of a tree
    'learning_rate': 0.1,  # Learning rate
    'subsample': 0.8,  # Subsample ratio
    'colsample_bytree': 0.8,  # Subsample ratio of columns
    'device': 'cuda',
}

bst = xgb.train(params, dtrain, num_boost_round=1000)

y_pred_prob = bst.predict(dtest)
diff_test = pd.read_csv(test_processed_diff)
y_test = diff_test.to_numpy()

kl_divergence = np.sum(rel_entr(y_test, y_pred_prob))
print(f'KL Divergence: {kl_divergence:.4f}')