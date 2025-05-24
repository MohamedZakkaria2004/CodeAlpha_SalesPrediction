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

### 2. **Exploratory Data Analysis (EDA)**
- Generate pair plots to observe feature relationships
- Create a heatmap to visualize correlation between features
- Understand the influence of advertising mediums on sales

### 3. **Model Building**
- Split the dataset into training and testing sets
- Train a **Linear Regression** model using `scikit-learn`
- Predict sales on test data

### 4. **Model Evaluation**
- Evaluate performance using:
  - R² Score
  - RMSE (Root Mean Squared Error)
- Visualize predicted vs actual sales with scatter plots

### 5. **Advertising Impact Analysis**
- Examine the model’s coefficients to determine which advertising channel contributes most to sales
- Use this insight to guide marketing decisions

---

## 📊 Results

- The linear regression model provides a clear understanding of how advertising investments affect sales.
- TV and Radio generally have a stronger influence on sales compared to Newspaper, based on model coefficients.

---

## 📎 Dependencies

Make sure the following Python libraries are installed:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn

