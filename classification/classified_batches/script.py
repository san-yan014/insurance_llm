import pandas as pd

df = pd.read_excel('classified_batches/40_standard_20_rough_classified.xlsx')

print(df['publication'].value_counts())
