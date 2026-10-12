import pandas as pd

date_fotbal = {
    'Jucator': ['A', 'A', 'B', 'B'],
    'Tip_Meci': ['Amical', 'Oficial', 'Amical', 'Oficial'],
    'Goluri': [90, 1, 10, 15],
    'Suturi': [100, 10, 10, 100]
}

df = pd.DataFrame(date_fotbal)
df['Rata_Conversie'] = df['Goluri'] / df['Suturi']

print(df[['Jucator', 'Tip_Meci', 'Rata_Conversie']])

df_total = df.groupby('Jucator')[['Goluri', 'Suturi']].sum().reset_index()
df_total['Rata_Totala'] = df_total['Goluri'] / df_total['Suturi']

print("\n")
print(df_total[['Jucator', 'Rata_Totala']])