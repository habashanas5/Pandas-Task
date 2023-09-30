import pandas as pd
from sklearn.impute import SimpleImputer
df = pd.read_csv("path/to/covid-data.csv")
rows, cols = df.shape
print("Number of rows:", rows)
print("Number of columns:", cols)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("First 10 rows:")
print(df.head(10))

print("Last 5 rows:")
print(df.tail(5))
print(df.describe())
print(df.nunique())
columns_to_drop = ['new_deaths_smoothed', 'new_cases_per_million', 'total_cases_per_million']
df.drop(columns_to_drop, axis=1, inplace=True)
column_mapping = {
    'date': 'Date',
    'location': 'Country',
    'continent': 'Continent',
    'iso_code': 'ISO_code'
}
df.rename(columns=column_mapping, inplace=True)
continent_names = df['Continent'].unique().tolist()
print("Continent Names:", continent_names)
imputer = SimpleImputer(strategy='constant', fill_value=0)
df2 = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
columns_to_replace = ['total_cases', 'total_deaths', 'total_vaccinations']
df2[columns_to_replace] = df2[columns_to_replace].replace('missing_value', 0)
total_countries = df2[df2['total_deaths'] > 1000000]['Country'].nunique()
print("Total countries with total_deaths > 1000000:", total_countries)
total_dates = df2[df2['total_deaths'] > 1000000]['Date'].nunique()
print("Number of dates with total_deaths > 1000000:", total_dates)