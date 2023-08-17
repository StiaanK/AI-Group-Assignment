import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Preparing data
df = pd.read_csv('C:\\Users\\Stiaan\\Desktop\\Honeurs\\Sem 2\\ITRW626\\AI-Group-Assignment\\Phase1\\adult.csv', delimiter=';')

# Define feature columns and target column
cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country', 'income']

# Encode categorical features using LabelEncoder
label_encoders = {}
encoding_info = {}

for col in cols:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
        # Create a dictionary to store encoding information
        encoding_info[col] = {original_value: encoded_value for original_value, encoded_value in zip(le.classes_, le.transform(le.classes_))}
        
# Save the DataFrame with integer-encoded values to a new CSV file
df.to_csv('./adult_encoded.csv', index=False)

# Save encoding information to a text file
with open('./encoding_info.txt', 'w') as file:
    for col, info in encoding_info.items():
        file.write(f"Column: {col}\n")
        for original_value, encoded_value in info.items():
            file.write(f"   Original value: {original_value} => Encoded value: {encoded_value}\n")

