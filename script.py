import pandas as pd

survey = pd.read_csv("masculinity.csv")
pd.set_option('display.max_columns', None)
print(survey.head())
