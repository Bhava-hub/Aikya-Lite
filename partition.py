import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/creditcard.csv")

# Same split as baseline.py — same random_state guarantees identical test set
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Class"]
)

# Save the held-out global test set (used for fair evaluation later)
test_df.to_csv("data/global_test.csv", index=False)
print(f"Global test set: {test_df.shape}, fraud cases: {test_df['Class'].sum()}")

# Partition ONLY the training portion into non-IID bank shards
# Partition ONLY the training portion into non-IID bank shards
train_sorted = train_df.sort_values("Amount").reset_index(drop=True)

n_banks = 4
split_points = np.array_split(np.arange(len(train_sorted)), n_banks)

for i, idx in enumerate(split_points):
    shard = train_sorted.iloc[idx]
    shard.to_csv(f"data/bank_{i}.csv", index=False)
    fraud_count = shard['Class'].sum()
    total = len(shard)
    print(f"Bank {i}: {total} transactions, {fraud_count} fraud ({fraud_count/total*100:.3f}%)")