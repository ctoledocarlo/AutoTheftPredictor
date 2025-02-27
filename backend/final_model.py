import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler


# Data Exploration

# 1. Load data and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropiate

# Path for file
df = pd.read_csv("C:/DataWarehouse/Auto_Theft_Open_Data.csv")

# Some basic info about the Data Frame
print('Data Frame head:')
print(df.head())

print('Data Frame info:')
print(df.info())

print('Data Frame describe:')
print(df.describe())

# 2. Statistical assessments including means, averages, and correlations.

# Mean, Median, and Mode

# Report Year
print('Report Year Mean')
print(df['OCC_HOUR'].mean())

print('Report Year Median')
print(df['OCC_HOUR'].median())

print('Report Year Mode')
print(df['OCC_HOUR'].mode())

# Occurrence Hour (Time)
print('Occurrence Hour Mean')
print(df['OCC_HOUR'].mean())

print('Occurrence Hour Median')
print(df['OCC_HOUR'].median())

print('Occurrence Hour Mode')
print(df['OCC_HOUR'].mode())


# Correlation Matrix

# Had to create new df only with numeric columns
df_numeric = df.select_dtypes(include=[np.number])
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# 3. Missing Data Visualizations

# Look for missing values
print(df.isnull().sum())


# Drop rows with missing values
# We could replace the value instead of dropping, but since we have 60k+ rows,
# dropping 4 of them shouldn't affect the analysis
df_cleaned = df.dropna()
# Check that there are no missing values
print(df_cleaned.isnull().sum())


# 4. Graphs and Visualization

# Histogram for Report Year
plt.hist(df['REPORT_YEAR'], bins=11)
plt.xticks(range(2014, 2024))
plt.xlabel('Report Year')
plt.ylabel('Frequency')
plt.title('Histogram of Report Year')
plt.show()

# Histogram for Occurrence Hour
plt.hist(df['OCC_HOUR'], bins=24, range=(0, 25))
plt.xticks(range(0, 24))  # Set x-axis ticks for each hour
plt.xlabel('Occurrence Hour')
plt.ylabel('Frequency')
plt.title('Histogram of Occurrence Hour')
plt.show()

# Boxplot for Report Hour by Report Month
sns.boxplot(x='REPORT_MONTH', y='REPORT_HOUR', data=df)
plt.title('Box Plot of Report Hour by Report Month')
plt.show()

# Time Series Plot for Number of Reports per year
df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])
df.set_index('REPORT_DATE', inplace=True)
df.resample('M').size().plot()
plt.title('Number of Reports per Year')
plt.show()

# Count Plot for Division
sns.countplot(x='DIVISION', data=df)
plt.title('Distribution of Incidents by Division')
plt.show()


#------------------------------------------------------------


# Convert REPORT_MONTH from string to numeric (e.g., 'September' to 9)
if df_cleaned['REPORT_MONTH'].dtype == 'object':
    df_cleaned['REPORT_MONTH'] = pd.to_datetime(df_cleaned['REPORT_MONTH'], format='%B').dt.month

# Create cyclical features for REPORT_HOUR
df_cleaned['REPORT_HOUR_sin'] = np.sin(2 * np.pi * df_cleaned['REPORT_HOUR'] / 24)
df_cleaned['REPORT_HOUR_cos'] = np.cos(2 * np.pi * df_cleaned['REPORT_HOUR'] / 24)

# Regression Target
target_regression = 'OCC_HOUR'

# Classification Target (Risk Level)
hourly_counts = df_cleaned['OCC_HOUR'].value_counts().sort_index()
top_25_percent_threshold = hourly_counts.quantile(0.75)
high_risk_hours = hourly_counts[hourly_counts >= top_25_percent_threshold].index.tolist()
df_cleaned['Risk'] = df_cleaned['OCC_HOUR'].apply(lambda x: 1 if x in high_risk_hours else 0)

# Features for tasks
features = ['REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'LONG_WGS84', 'LAT_WGS84', 'REPORT_HOUR_sin', 'REPORT_HOUR_cos']

# Prepare datasets for regression
X_regression = df_cleaned[features]
y_regression = df_cleaned[target_regression]

# Prepare datasets for classification
X_classification = X_regression
y_classification = df_cleaned['Risk']

# Class Distribution Before Oversampling
print("Class Distribution Before Oversampling:")
print(y_classification.value_counts())

# Oversampling for imbalance
oversampler = RandomOverSampler(random_state=42)
X_resampled_reg, y_resampled_reg = oversampler.fit_resample(X_regression, y_regression)
X_resampled_cls, y_resampled_cls = oversampler.fit_resample(X_classification, y_classification)


# Class Distribution After Oversampling
print("\nClass Distribution After Oversampling:")
print(pd.Series(y_resampled_cls).value_counts())

# Split datasets into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_resampled_reg, y_resampled_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_resampled_cls, y_resampled_cls, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_cls, y_train_cls)
y_pred_cls = classifier.predict(X_test_cls)


print("Mean absolute error: ", mean_absolute_error(y_test_cls, y_pred_cls))
print("Root mean squared error: ", root_mean_squared_error(y_test_cls, y_pred_cls))

print("Accuracy: ", accuracy_score(y_test_cls, y_pred_cls))
print("Precision: ", precision_score(y_test_cls, y_pred_cls))
print("Recall: ", recall_score(y_test_cls, y_pred_cls))
print("F1 Score: ", f1_score(y_test_cls, y_pred_cls))
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls))

with open("random_forest_classifier.pkl", "wb") as file:
    pickle.dump(classifier, file)

print("random_forest_classifier saved.")