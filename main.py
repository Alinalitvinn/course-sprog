import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import os

options = [
    "Aave", 
    "BinanceCoin", 
    "Bitcoin", 
    "Cardano", 
    "ChainLink", 
    "Cosmos", 
    "CryptocomCoin", 
    "Dogecoin", 
    "EOS", 
    "Ethereum", 
    "Iota", 
    "Litecoin", 
    "Monero", 
    "NEM", 
    "Polkadot", 
    "Solana", 
    "Stellar", 
    "Tether", 
    "Tron", 
    "Uniswap", 
    "USDCoin", 
    "WrappedBitcoin", 
    "XRP"
]

def formating_data(options, year):
    result = pd.DataFrame()
    filename = f'format_data/cryptocurrency_data_by_{year}_year.csv'

    for option in options:
        df = pd.read_csv(f'csv_data/coin_{option}.csv')
        
        # фільтруємо по року
        mask = df['Date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year) == year
        filtered_df = df[mask]
        
        result = pd.concat([result, filtered_df])
        

    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping!")
    else:
        result.to_csv(f'format_data/cryptocurrency_data_by_{year}_year.csv', index=False)

formating_data(options=options, year=2019)
formating_data(options=options, year=2020)
formating_data(options=options, year=2021)

# Завдання 2
year_of_data = int(input("Enter year for work with data: "))
dataframe = pd.read_csv(f'format_data/cryptocurrency_data_by_{year_of_data}_year.csv', index_col='Name')

is_null = dataframe.isnull().values.any()
null_list = dataframe.isnull().sum()
print(f"Чи існують пусті значення: {is_null}.\nСписок до чистики: \n{null_list}")

cleaned_df = dataframe.ffill()
print(f"Список після чистки:\n{cleaned_df.isnull().sum()}")

# Завдання 3
dataframe = pd.read_csv(f'format_data/cryptocurrency_data_by_{year_of_data}_year.csv', index_col='Name')
dataframe.head(20)


# Завдання 4
selected_dataframe = dataframe.iloc[:10, 3:7]
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: green' if v else '' for v in is_max]

def highlight_min(s):  
    is_min = s == s.min()
    return ['background-color: red' if v else '' for v in is_min]

selected_dataframe = selected_dataframe.reset_index(drop=True) 
styled_df = selected_dataframe.style.apply(highlight_max).apply(highlight_min)
styled_df

# Завдання 4: додатково
selected_data = dataframe.iloc[:, 3:7]

mean = selected_data.mean()
std = selected_data.std()
variance = np.var(selected_data)
standardized = (selected_data - mean) / std

print(f"Математичне сподівання для вибраних даних:\n{mean}")
print(f"\nДисперсія для вибраних даних:\n{variance}")
print(f"\nСтандартне відхилення для вибраних даних:\n{std}")
print(f"\nСтандартизовані вибрані дані:\n{standardized.iloc[:10]}")

# Завдання 5
# Heatmap
plt.figure(figsize=(10, 6))
heat_data = dataframe.iloc[:20, 3:7]
sub_data = heat_data.to_numpy()

sns.heatmap(data=sub_data, cmap='crest', vmin=sub_data.min(), vmax=sub_data.max(), annot=True, linewidth=.5, fmt=".2f")

plt.ylabel('Ціна, $')
plt.title('Різні позиції цін на валюти')
plt.show()

# LinePlot
plt.figure(figsize=(10, 6))

data = dataframe
data.reset_index(inplace=True)
data = data[data['Name'] == 'Dogecoin']
data = data.iloc[:30]

sns.lineplot(data=data, x='Date', y='High', style="Name", markers=True, dashes=False)
plt.xlabel('Дата')
plt.ylabel('Найвища ціна, $')
plt.xticks(rotation=90)
plt.title('Найивща ціна ща день')
plt.show()

# BarPlot для низьких цін
plt.figure(figsize=(10, 6))

crypto_df = dataframe[~dataframe['Name'].isin(['Bitcoin', 'Wrapped Bitcoin', 'Ethereum'])]

ax = sns.barplot(data=crypto_df, x='Name', y='Low')
ax.bar_label(ax.containers[0], fontsize=10)

plt.xlabel('Дата')
plt.ylabel('Найнижча ціна, $')
plt.xticks(rotation=90)
plt.title('Найнижча ціна за день')
plt.show()

# BarPlot для вискоих цін
plt.figure(figsize=(10, 6))

btc_df = dataframe[dataframe['Name'].isin(['Bitcoin', 'Wrapped Bitcoin', 'Ethereum'])]

ax = sns.barplot(data=btc_df, x='Name', y='Low')
ax.bar_label(ax.containers[0], fontsize=10)

plt.xlabel('Дата')
plt.ylabel('Найнижча ціна, $')
plt.xticks(rotation=90)
plt.title('Найнижча ціна за день')
plt.show()

# HistPlot
plt.figure(figsize=(10, 6))

selected_data = dataframe[dataframe['Date'] == '2020-10-05 23:59:59']
crypto_df = selected_data[~selected_data['Name'].isin(['Bitcoin', 'Wrapped Bitcoin', 'Ethereum'])]
sns.histplot(data=crypto_df, x='Close', bins=30, hue='Name')
    
plt.xlabel('Ціна на укладанні угоди')
plt.xticks(rotation=90)
plt.title('Гістограма цін на укладанні угоди')
plt.show()

# Завдання 6
plt.figure(figsize=(10, 6))
fig, ax = plt.subplots()

data = dataframe
data.reset_index(inplace=True)
btc_df = data[data['Name'].isin(['Bitcoin', 'Wrapped Bitcoin', 'Ethereum'])]

for column in ['High','Low','Open','Close']:
    ax.plot(btc_df['Name'], btc_df[column], label=column)

ax.legend()

y_values = btc_df[data.columns[6:10]].mean().values
x_values = btc_df['Name'].unique()

for x,y in zip(x_values, y_values):  
    ax.annotate(f"{y:.2f}", xy=(x,y), xytext=(5, 5), textcoords='offset points') 
        
ax.autoscale()
plt.xticks(rotation=90)
plt.show()

# Завдання 6: додатково побудувати PairPlot
data = dataframe.iloc[:400, 5:9]
sns.pairplot(data=data)
plt.show()

# Завдання 7
pearson_corr = dataframe.iloc[:, 5:9].corr(method='pearson') 
print(f"Коефіцієнт пірсона:\n{pearson_corr}")

x_col = dataframe.columns[6]
y_cols = pearson_corr.idxmax().index

sns.relplot(data=dataframe, x=x_col, y=y_cols[0], hue=y_cols[1], kind="scatter")
    
plt.xlabel(x_col)
plt.ylabel(y_cols[0])
plt.title('Relplot')
plt.show()


# Завдання 8
dataframe = pd.read_csv(f'format_data/cryptocurrency_data_by_{year_of_data}_year.csv')
print(f"До заміни:\n{dataframe.isnull().sum()}")

dataframe = dataframe.ffill()
print(f"Після заміни:\n{dataframe.isnull().sum()}")

dataframe.head(20)

# Завдання 9
df_with_10 = dataframe.drop(['Name'], axis=1)
df_without_10 = df_with_10.iloc[np.random.choice(df_with_10.shape[0], round(df_with_10.shape[0] * 0.9))]

numeric_columns = df_with_10.select_dtypes(include=['float64']).columns
df_with_10_numeric = df_with_10[numeric_columns]
df_without_10_numeric = df_without_10[numeric_columns]

plt.scatter("High", "Low", data=df_without_10_numeric.iloc[:100, :])
plt.xlabel("High")
plt.ylabel("Low")
plt.legend()
plt.show()