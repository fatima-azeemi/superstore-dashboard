import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Superstore Sales Intelligence System",
    layout="wide"
)

# Load data
df = pd.read_csv("Superstore.csv", encoding="latin1")

# ---------------------------
# Feature Engineering
# ---------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Month"] = df["Order Date"].dt.month
df["Year"] = df["Order Date"].dt.year

# ---------------------------
# Title
# ---------------------------
st.title("Superstore Sales Intelligence System")
st.title("Fatima Azeemi")
st.markdown("---")

# ---------------------------
# KPIs
# ---------------------------
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"{total_sales:,.0f}")
col2.metric("Total Profit", f"{total_profit:,.0f}")
col3.metric("Total Orders", total_orders)
col4.metric("Total Customers", total_customers)

st.markdown("---")

# ---------------------------
# Filters
# ---------------------------
st.sidebar.header("Filters")

region = st.sidebar.multiselect("Region", df["Region"].unique(), df["Region"].unique())
category = st.sidebar.multiselect("Category", df["Category"].unique(), df["Category"].unique())
segment = st.sidebar.multiselect("Segment", df["Segment"].unique(), df["Segment"].unique())

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Segment"].isin(segment))
]

# ---------------------------
# Monthly Sales Trend (Plotly)
# ---------------------------
st.subheader("Monthly Sales Trend")

monthly = filtered_df.groupby("Month")["Sales"].sum().reset_index()

fig = px.line(monthly, x="Month", y="Sales", markers=True)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Category Profit
# ---------------------------
st.subheader("Category Wise Profit")

cat_profit = filtered_df.groupby("Category")["Profit"].sum().reset_index()

fig = px.bar(cat_profit, x="Category", y="Profit", color="Category")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Top Products
# ---------------------------
st.subheader("Top 10 Products by Sales")

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(top_products, x="Sales", y="Product Name", orientation="h")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Discount vs Profit
# ---------------------------
st.subheader("Discount vs Profit Relationship")

fig = px.scatter(filtered_df, x="Discount", y="Profit", color="Category")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Model Load
# ---------------------------
model = joblib.load("profit_model.pkl")
columns = joblib.load("model_columns.pkl")

# ---------------------------
# Prediction Section
# ---------------------------
st.subheader("Profit Prediction System")

col1, col2, col3 = st.columns(3)

sales = col1.number_input("Sales", min_value=0.0)
quantity = col2.number_input("Quantity", min_value=1)
discount = col3.number_input("Discount", min_value=0.0, max_value=1.0)

category_input = st.selectbox("Category", df["Category"].unique())
sub_category = st.selectbox("Sub-Category", df["Sub-Category"].unique())
region_input = st.selectbox("Region", df["Region"].unique())
segment_input = st.selectbox("Segment", df["Segment"].unique())
ship_mode = st.selectbox("Ship Mode", df["Ship Mode"].unique())

month = st.slider("Month", 1, 12, 6)
year = st.number_input("Year", min_value=2015, max_value=2030, value=2024)

if st.button("Predict Profit"):

    input_df = pd.DataFrame([{
        "Sales": sales,
        "Quantity": quantity,
        "Discount": discount,
        "Month": month,
        "Year": year,
        "Category": category_input,
        "Sub-Category": sub_category,
        "Region": region_input,
        "Segment": segment_input,
        "Ship Mode": ship_mode
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Profit: {prediction:.2f}")