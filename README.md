# 🚗 Predicting Automobile Theft Likelihood

A machine learning project that predicts high-risk periods and locations for automobile theft in Toronto using police data from 2014-2024.

## 📋 Project Overview

This project develops a predictive model for automobile theft likelihood in Toronto by analyzing 66,038 records collected by the Toronto Police Service over a decade (2014-2024). Using machine learning techniques, we identify patterns in theft occurrences to help law enforcement and the public implement preventive measures.

### 🔍 Key Findings

- Automobile thefts peak in the afternoon and late-night hours
- Temporal and location-based features are critical predictors
- Classification model achieved 72% accuracy in identifying high-risk hours
- Random Forest algorithms provide the best performance for both regression and classification tasks

## 📊 Data Exploration

### Dataset Details

- **Source**: Toronto Police Service, 2014-2024
- **Records**: 66,038
- **Features**: 31 columns including temporal, geospatial, and categorical variables
- **Reference**: [Google Sheets Data Source](https://docs.google.com/spreadsheets/d/1hH659gOyz7XR_XTa72J5WrFFmuiX0WvR3j21IQ525Oc/edit?usp=sharing)

### Key Insights

- **Temporal Patterns**: Theft occurrences peak at midnight (12 AM) and show a secondary peak around 3 PM (15:00)
- **Yearly Trends**: Reports steadily increased over time, peaking around 2022
- **Divisional Analysis**: Specific police divisions (D31, D32, D41) report significantly higher incidents
- **Reporting Consistency**: Median report time remains around 3 PM across all months

## 🛠️ Methodology

### Feature Engineering

- **Cyclical Encoding**: Applied to temporal features like `REPORT_HOUR`
- **One-Hot Encoding**: Used for categorical features (`DIVISION`, `LOCATION_TYPE`, `PREMISES_TYPE`)
- **Feature Selection**: Identified key predictors including `REPORT_YEAR`, `REPORT_HOUR`, and `DIVISION`

### Model Development

- **Data Split**: 80% training, 20% testing
- **Oversampling**: Applied RandomOverSampler to address class imbalance
- **Risk Definition**:
  - High-Risk (Class 1): Theft hours in the top 25% of frequencies
  - Low-Risk (Class 0): Theft hours in the remaining 75%

## 📈 Results

### Classification Performance (Random Forest)

- **Accuracy**: 72.03%
- **Precision (High-Risk)**: 70.08%
- **Recall (High-Risk)**: 75%
- **F1 Score**: 72.46%

### Class-Specific Metrics

- **Low Risk (Class 0)**:
  - Precision: 74%
  - Recall: 69%
  - F1 Score: 72%
- **High Risk (Class 1)**:
  - Precision: 70%
  - Recall: 75%
  - F1 Score: 72%

## 💡 Applications

This model can be used by:

- **Law Enforcement**: To optimize patrol schedules and resource allocation
- **Vehicle Owners**: To increase awareness during high-risk periods
- **Insurance Companies**: To develop data-driven risk assessment models
- **Urban Planners**: To design safer parking infrastructure

## 🔧 Technologies Used

- **Programming**: Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, Random Forest
- **Data Visualization**: Matplotlib, Seaborn

## 📝 Future Work

- Incorporate additional features like weather data and socioeconomic indicators
- Develop a real-time prediction system
- Expand the model to other metropolitan areas
- Create a user-friendly web interface for public access to risk predictions

## 📚 References

- Toronto Police Service Open Data Portal
- Machine Learning for Crime Prevention: A Review of Methods and Applications
- Temporal Pattern Analysis of Vehicle Theft: Case Studies and Preventive Strategies

---

⭐ Star this repository if you find it useful!
