# Data Preprocessing and Encoding

This code is designed to preprocess a dataset, handle missing values, and perform categorical feature encoding using LabelEncoder. The dataset used is taken from the `adult.csv` file, which presumably contains information about individuals and their socio-economic attributes.

## Table of Contents

- [Objective](#objective)
- [Steps](#steps)
- [Code Explanation](#code-explanation)
- [Dependencies](#dependencies)

## Objective

The main objectives of this code are:

1. Load the dataset from the `adult.csv` file.
2. Remove rows with missing values.
3. Save the cleaned dataset into a new CSV file named `adult_dropped_missing.csv`.
4. Encode categorical features using the `LabelEncoder` from scikit-learn.
5. Print out the mapping of original categorical values to their integer-encoded counterparts.
6. Save the DataFrame with integer-encoded values into a new CSV file named `adult_encoded.csv`.

## Steps

1. **Load and Clean Data:**
   - The code starts by loading the dataset from the `adult.csv` file using the Pandas library.
   - It then creates a new DataFrame `df_dropped` by removing rows with missing values using the `dropna()` function.

2. **Save Cleaned Data:**
   - The cleaned DataFrame `df_dropped` is saved into a new CSV file named `adult_dropped_missing.csv` using the `to_csv()` function.

3. **Define Feature Columns and Target Column:**
   - A list of feature column names (`cols`) is defined. This list includes columns like age, workclass, education, etc.
   - The target column name (`class`) is also included in the list.

4. **Encode Categorical Features:**
   - The code then enters a loop that iterates through each column in the `cols` list.
   - If the column's data type is `'object'`, it indicates that the column contains categorical data that needs to be encoded.
   - For each categorical column, the code uses the `LabelEncoder` from scikit-learn to transform the original categorical values into integer-encoded values.
   - The mapping of original values to encoded values is printed for each column.

5. **Save Encoded Data:**
   - Finally, the DataFrame with integer-encoded values is saved into a new CSV file named `adult_encoded.csv`.

## Code Explanation

The code aims to ensure that the dataset is cleaned from missing values and that categorical features are transformed into a format suitable for machine learning algorithms. It follows these steps:
1. Load the dataset and drop rows with missing values.
2. Save the cleaned dataset into a new CSV file.
3. Encode categorical features using the LabelEncoder.
4. Print out the original-to-encoded value mappings for each categorical column.
5. Save the fully preprocessed dataset with encoded features into another CSV file.

## Dependencies

- Python 3.x
- pandas
- scikit-learn

---

Feel free to tailor this README according to your needs and include any additional information you find relevant.