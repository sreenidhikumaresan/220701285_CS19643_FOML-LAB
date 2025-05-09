import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('icd_sample_data.csv')

# Display data info
print("Data Info:")
print(df.info())

# Display first few rows
print("\nSample Data:")
print(df.head())

# Fix column name if needed (in case of trailing spaces or case mismatch)
df.columns = df.columns.str.strip().str.lower()

# Show number of unique ICD codes
try:
    print("\nNumber of Unique ICD Codes:")
    print(df['icd_code'].nunique())
except KeyError:
    print("Column 'icd_code' not found. Available columns:", df.columns)

# ICD Code Frequency
try:
    icd_counts = df['icd_code'].value_counts()

    # Print top 5 most frequent codes
    print("\nTop ICD Codes:")
    print(icd_counts.head())

    # Plot the frequency
    plt.figure(figsize=(8, 6))
    icd_counts.plot(kind='bar')
    plt.title('ICD Code Frequency')
    plt.xlabel('ICD Code')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

except KeyError:
    print("Cannot visualize: Column 'icd_code' not found.")
