import pandas as pd
X_train = pd.read_csv('data/processed/train_processed_sample.csv', low_memory=False)
print(X_train)
print("Hellow")