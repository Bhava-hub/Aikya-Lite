import pandas as pd
df = pd.read_csv("data/creditcard.csv")
print(df.shape)
print(df['Class'].value_counts())