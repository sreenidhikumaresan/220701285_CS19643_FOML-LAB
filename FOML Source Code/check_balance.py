import pandas as pd

# Load your dataset
df = pd.read_csv('icd_sample_data_full_enhanced.csv')  # use your correct dataset name here

# Show number of samples for each ICD code
print("\n✅ Sample counts per ICD Code:")
print(df['icd_code'].value_counts())

# Optional: Map ICD code to human-readable condition if needed
# Example Mapping (Add more if you want)
icd_mapping = {
    'R50.9': 'Fever',
    'R05': 'Cough',
    'I10': 'Hypertension',
    'E11.9': 'Type 2 Diabetes',
    'J45.909': 'Asthma',
    'S02.9': 'Fracture of skull',
    'S52.9': 'Fracture of arm',
    'G43.909': 'Migraine',
    'R07.9': 'Chest pain',
    'L29.9': 'Skin rash',
    'F41.9': 'Anxiety disorder',
    'J18.9': 'Pneumonia',
    'I63.9': 'Stroke',
    'N18.9': 'Kidney Disease',
    'K25.9': 'Gastric Ulcer',
    'O26.9': 'Pregnancy complication',
    'M19.90': 'Arthritis',
    'U07.1': 'COVID-19 infection',
    'H66.9': 'Ear Infection',
    'J32.9': 'Sinusitis',
    'J00': 'Common Cold',
    'R17': 'Jaundice',
    'A09': 'Diarrhea',
    'R11': 'Vomiting'
}

print("\n✅ Human-readable sample counts:\n")
for icd_code, count in df['icd_code'].value_counts().items():
    condition = icd_mapping.get(icd_code, 'Unknown Condition')
    print(f"{condition} ({icd_code}): {count} samples")
