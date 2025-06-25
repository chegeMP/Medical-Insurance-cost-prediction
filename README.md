# 🏥💰 Insurance Cost Predictor

A machine learning model that predicts medical insurance costs based on personal demographics and lifestyle factors. This project demonstrates practical application of ML in the healthcare finance sector.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)

## 🎯 Problem Statement

Medical insurance costs vary significantly based on individual factors. This project helps:
- **Insurance Companies**: Assess risk and set appropriate premiums
- **Individuals**: Estimate their potential insurance costs
- **Healthcare Analysts**: Understand cost-driving factors

## 📊 Features & Dataset

The model analyzes the following factors:
- **Age**: Primary demographic factor
- **Gender**: Male/Female classification
- **BMI**: Body Mass Index health indicator
- **Children**: Number of dependents
- **Smoker Status**: Major cost determinant
- **Region**: Geographic location impact

## 🤖 Model Performance

- **Algorithm**: [Specify your algorithm - Linear Regression/Random Forest/etc.]
- **Accuracy**: [Add your R² score]
- **Mean Absolute Error**: [Add your MAE]
- **Training Data**: [Specify dataset size]

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Usage
```python
# Load the model
import pickle
import pandas as pd

# Example prediction
sample_data = {
    'age': 25,
    'sex': 'female',
    'bmi': 22.5,
    'children': 0,
    'smoker': 'no',
    'region': 'northeast'
}

# Predict cost
predicted_cost = model.predict([sample_data])
print(f"Estimated Insurance Cost: ${predicted_cost[0]:.2f}")
```

## 📈 Key Insights

- **Smoking** is the strongest predictor of high insurance costs
- **Age** shows positive correlation with insurance premiums
- **BMI** significantly impacts cost predictions
- **Regional variations** affect pricing strategies

## 🔧 Technical Implementation

```python
# Feature engineering pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Data preprocessing
le = LabelEncoder()
X['sex'] = le.fit_transform(X['sex'])
X['smoker'] = le.fit_transform(X['smoker'])
X['region'] = le.fit_transform(X['region'])

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## 📊 Visualizations

The project includes:
- Cost distribution analysis
- Feature correlation heatmaps
- Age vs. Cost scatter plots
- Regional cost comparisons

## 🌍 Real-World Applications

- **Insurance Underwriting**: Automate premium calculations
- **Healthcare Planning**: Budget estimation for individuals
- **Policy Analysis**: Understand demographic cost patterns
- **Risk Assessment**: Identify high-cost customer segments

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional feature engineering
- Model ensemble methods
- Web interface development
- API endpoint creation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Connect

Built with ❤️ by [ChegeMP](https://github.com/chegeMP)
- Passionate about AI applications in healthcare
- Based in Nairobi, Kenya

---

*This project demonstrates practical machine learning application in insurance sector, showcasing data preprocessing, model training, and real-world problem solving.*
