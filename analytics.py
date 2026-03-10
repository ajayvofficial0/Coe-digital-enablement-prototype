import pandas as pd

# Load dataset
df = pd.read_csv("data/coe_initiatives.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset shape before cleaning:")
print(df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)

# Convert date columns
df["Start Date"] = pd.to_datetime(df["Start Date"])
df["End Date"] = pd.to_datetime(df["End Date"])

# Create KPI achievement percentage
df["KPI_Achievement_%"] = (df["KPI Achieved"] / df["KPI Target"]) * 100

# Validate status values
print("\nUnique status values:")
print(df["Status"].unique())

# EDA
print("\nStatus Distribution:")
print(df["Status"].value_counts())

print("\nAverage KPI Achievement %:")
print(round(df["KPI_Achievement_%"].mean(), 2))

print("\nTop 3 Initiatives by Business Benefit:")
top3 = df.sort_values("Business Benefit", ascending=False).head(3)
print(top3[["Initiative Name", "Business Benefit", "KPI_Achievement_%"]])

# Save cleaned dataset
df.to_csv("data/cleaned_coe_initiatives.csv", index=False)

print("\nCleaned dataset saved successfully as data/cleaned_coe_initiatives.csv")