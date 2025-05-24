# CodeAlpha_SalesPrediction
Predicts product sales using advertising data through linear regression &amp; analyzes the impact of TV, Radio, and Newspaper ads on sales performance.

## 📁 Dataset

**File:** `Advertising.csv`  
**Features:**
- `TV`: Advertising spend on TV (in thousands)
- `Radio`: Advertising spend on Radio (in thousands)
- `Newspaper`: Advertising spend on Newspaper (in thousands)
- `Sales`: Actual sales (target variable)

---

## 🔧 Project Workflow

### 1. **Data Preparation**
- Load the dataset using `pandas`
- Clean missing or duplicate values
- Select features for modeling

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df = pd.read_csv('Advertising.csv')

# Data Cleaning & Preprocessing
df.dropna(inplace=True)

### 2. **Exploratory Data Analysis (EDA)**
- Generate pair plots to observe feature relationships
- Create a heatmap to visualize correlation between features
- Understand the influence of advertising mediums on sales

# Exploratory Data Analysis
sns.pairplot(df)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title("Correlation Heatmap")
plt.show()

### 3. **Model Building**
- Split the dataset into training and testing sets
- Train a **Linear Regression** model using `scikit-learn`
- Predict sales on test data

### 4. Feature Selection

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 5. **Model Evaluation**
- Evaluate performance using:
  - R² Score
  - RMSE (Root Mean Squared Error)
- Visualize predicted vs actual sales with scatter plots

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

### 6. **Advertising Impact Analysis**
- Examine the model’s coefficients to determine which advertising channel contributes most to sales
- Use this insight to guide marketing decisions

# Advertising Impact
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()


## 📊 Results

- The linear regression model provides a clear understanding of how advertising investments affect sales.
- TV and Radio generally have a stronger influence on sales compared to Newspaper, based on model coefficients.


