# Credit_Card_Transaction_analysis_and_prediction
# Credit Card Spend Analysis & Forecasting

Dataset: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset

## 1. Introduction
Credit card transactions generate vast amounts of data, providing valuable insights into customer spending patterns. Financial institutions and businesses can leverage this data to forecast future spending trends, optimize marketing strategies, and enhance customer experience.

This project aims to analyze customer spending behavior based on transaction categories (e.g., food, travel, shopping) and build a time-series forecasting model (Facebook Prophet) to predict future spending trends. The model will be deployed as a Streamlit web app, allowing users to input parameters and get spending forecasts.

## 2. Problem Definition
Financial institutions and businesses often struggle to:
- Identify spending trends among customers across various categories.
- Predict future spending behavior to improve budgeting and marketing strategies.
- Personalize financial services such as credit limits and cashback offers.

## 3. Dataset Description
**Dataset Name:** Credit Card Transactions Dataset  
**Data Source:**
- Publicly available datasets from Kaggle/UCI Machine Learning Repository.

### Dataset Features:
| Column Name | Description |
|-------------|-------------|
| Transaction_ID | Unique identifier for each transaction |
| Customer_ID | Unique ID for each customer |
| Transaction_Date | Date of transaction |
| Transaction_Amount | Amount spent in each transaction |
| Category | Spending category (e.g., Food, Travel, Shopping, Entertainment) |
| Merchant | Name of the merchant/store |
| Payment_Method | Mode of payment (Credit, Debit, Online, POS) |
| Location | City or country of transaction |
| Balance_After_Transaction | Remaining balance after transaction |

**Target Variable:**
- Transaction_Amount (Used for predicting future spending patterns).

**Potential Data Sources:**
- Kaggle: Credit Card Transactions Dataset
- UCI ML Repository
- Generate synthetic data using Python (Faker, NumPy, Pandas)

## 4. Exploratory Data Analysis (EDA)
### 4.1 Basic EDA Questions (Beginner-Level)
1. What are the top spending categories by transaction volume and amount?
2. How does spending vary over time (daily, weekly, monthly trends)?
3. What are the peak spending hours in a day?
4. What is the most common payment method used?
5. Which merchants have the highest transactions?

### 4.2 Intermediate EDA Questions
1. Are there seasonal trends in spending across different categories?
2. How do different customer segments (high spenders vs. low spenders) behave?
3. What is the distribution of transaction amounts (e.g., histogram, boxplot analysis)?
4. Are there any correlations between spending behavior and location?
5. How does spending behavior change before and after payday?

### 4.3 Advanced EDA Questions
1. Can we detect outliers in spending behavior using anomaly detection?
2. Are there clusters of customers based on spending habits? (Use K-Means, DBSCAN)
3. Can we use association rule mining (Apriori, FP-Growth) to find patterns in purchases?
4. How does spending behavior correlate with economic factors (e.g., inflation, interest rates)?
5. Can we use NLP on transaction descriptions to classify transactions more effectively?

## 5. Machine Learning Model – Facebook Prophet
### Model Selection:
We will use Facebook Prophet, a powerful time-series forecasting tool designed for financial and business data. It is robust to missing data and seasonal variations.

### Steps to Build the Model:
1. **Prepare Data**
   - Convert Transaction_Date to a time-series format.
   - Aggregate transactions by day/month for each spending category.
2. **Train Model**
   - Use Transaction_Amount as the target variable.
   - Include external regressors (e.g., holiday effects, economic factors).
3. **Evaluate Model Performance**
   - Use metrics like MAE (Mean Absolute Error) and RMSE (Root Mean Square Error).
   - Compare Prophet’s forecast with actual data.

## 6. Deployment – Streamlit Web App
### Web App Features:
🎯 **User Input Panel** – Select customer ID, spending category, and time range.  
📊 **Interactive Data Visualization** – View historical spending trends with graphs.  
🔮 **Future Forecasting** – Predict spending trends for the next 3-6 months.  

## 7. Expected Outcomes
🔹 Improved understanding of customer spending patterns.  
🔹 Accurate predictions of future credit card spending.  
🔹 A user-friendly dashboard for financial institutions & individuals.  
🔹 Potential integration with banks for real-time forecasting & budget recommendations.

