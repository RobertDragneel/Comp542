import pandas as pd

# Load your dataset
df = pd.read_csv('your_sunspot_data.csv', parse_dates=['Date'], index_col='Date')

# Rename the column for convenience
df.rename(columns={'Monthly Mean Total Sunspot Number': 'Sunspots'}, inplace=True)

# Check the data
print(df.head())