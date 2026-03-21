import pandas as pd

# Step 1 — peek at the 3 small files
# Just reading 5 rows to see column names

patients   = pd.read_csv('patients.csv', nrows=5)
admissions = pd.read_csv('admissions.csv', nrows=5)
icustays   = pd.read_csv('icustays.csv', nrows=5)

print("--- patients columns ---")
print(list(patients.columns))

print("\n--- admissions columns ---")
print(list(admissions.columns))

print("\n--- icustays columns ---")
print(list(icustays.columns))