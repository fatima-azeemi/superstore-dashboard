# Superstore Sales Intelligence System

## Author
Fatima Azeemi

---

## Project Overview
The Superstore Sales Intelligence System is a complete Data Analytics and Machine Learning project designed to analyze sales performance, profit trends, customer behavior, and business insights using real-world retail data.

The system also includes a Machine Learning model to predict profit based on key business features.

---

## Business Objective
The goal of this project is to help businesses make data-driven decisions by identifying:
- Most profitable products
- High-performing regions
- Customer segment behavior
- Impact of discount on profit
- Future profit prediction using ML models

---

## Dataset Information
**Dataset Name:** Superstore Dataset  

### Columns:
- Row ID
- Order ID
- Order Date
- Ship Date
- Ship Mode
- Customer ID
- Customer Name
- Segment
- Country
- City
- State
- Region
- Product ID
- Category
- Sub-Category
- Product Name
- Sales
- Quantity
- Discount
- Profit

---

## Project Workflow

### 1. Data Cleaning
- Removed duplicates
- Handled missing values
- Converted date columns to datetime format

### 2. Feature Engineering
- Month extracted from Order Date
- Year extracted from Order Date
- Profit Margin calculated

### 3. Exploratory Data Analysis (EDA)
- Monthly Sales Trends
- Yearly Sales Performance
- Top 10 Products by Sales
- Top 10 Products by Profit
- Category-wise analysis
- Region-wise analysis
- Discount vs Profit relationship

### 4. Machine Learning Models
The following models were trained:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### 5. Model Evaluation
Models were evaluated using:
- R² Score
- RMSE (Root Mean Squared Error)

Best model selected based on test performance and generalization.

---

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Plotly

---

## Streamlit Dashboard Features
- KPI Cards (Sales, Profit, Orders, Customers)
- Interactive Filters (Region, Category, Segment)
- Sales Trend Visualization
- Profit Analysis Charts
- Discount vs Profit Analysis
- Machine Learning-based Profit Prediction

---

## Machine Learning Target
**Target Variable:** Profit  

### Input Features:
- Sales
- Quantity
- Discount
- Month
- Year
- Category
- Sub-Category
- Region
- Segment
- Ship Mode

---

## Model Deployment
The project is deployed using Streamlit and provides an interactive web-based dashboard for business intelligence.

---

## How to Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py