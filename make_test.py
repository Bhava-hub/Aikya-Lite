import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/creditcard.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Class"])
test_df.to_csv("data/global_test.csv", index=False)
print("Test set saved:", test_df.shape, "fraud cases:", test_df["Class"].sum())