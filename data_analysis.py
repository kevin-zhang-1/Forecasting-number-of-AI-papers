import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv('arxiv_cs_ai_monthly_counts.csv', parse_dates=['Month'])
df.columns = ['Month', 'Count']  # Ensure consistent column names

# Basic statistics
print("Key Statistics:")
print(f"Total Papers: {df['Count'].sum()}")
print(f"Average Monthly Papers: {df['Count'].mean():.1f}")
print(f"Peak Month: {df.loc[df['Count'].idxmax(), 'Month'].strftime('%Y-%m')} ({df['Count'].max()} papers)")
print(f"Median Monthly Papers: {df['Count'].median()}")

plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Month', y='Count', color='#2ecc71')
plt.title('Monthly cs.AI Papers on arXiv (1995–2023)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Papers')
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_trend.png', dpi=300)
plt.show()

# Aggregate by year
df_year = df.groupby(df['Month'].dt.year)['Count'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_year, x='Month', y='Count', palette='viridis')
plt.title('Yearly cs.AI Papers on arXiv', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Total Papers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('yearly_trend.png', dpi=300)
plt.show()

df['YoY Growth (%)'] = df['Count'].pct_change(periods=12) * 100  # Year-over-year
df['MoM Growth (%)'] = df['Count'].pct_change() * 100            # Month-over-month

df['Year'] = df['Month'].dt.year
df['MonthNumber'] = df['Month'].dt.month

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='MonthNumber', y='Count', palette='Set3')
plt.title('Monthly Distribution of Papers (1995–2023)')
plt.xlabel('Month')
plt.ylabel('Papers')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.savefig('monthly_distribution.png', dpi=300)
plt.show()