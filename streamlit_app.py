import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("🧮 Sales Forecasting Dashboard")

st.write(
    """
    Upload your **Sales Data** Excel file (same format as your current dataset).
    The app will show trends and predict sales for the next month using Machine Learning (XGBoost).
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload your sales data (xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if 'DATE' not in df.columns or 'SALES AMT' not in df.columns:
        st.error("File must have columns 'DATE' and 'SALES AMT'")
    else:
        df['DATE'] = pd.to_datetime(df['DATE'])
        # Show quick stats
        st.write("First 5 rows:", df.head())
        st.info(f"Rows: {len(df):,}, Columns: {df.columns.tolist()}")

        # Total monthly sales
        monthly = df.groupby(df['DATE'].dt.to_period('M'))['SALES AMT'].sum().reset_index()
        monthly['DATE'] = monthly['DATE'].dt.to_timestamp()

        # Add features
        monthly['month'] = monthly['DATE'].dt.month
        monthly['lag1'] = monthly['SALES AMT'].shift(1)
        monthly['rolling3'] = monthly['SALES AMT'].shift(1).rolling(3).mean()
        monthly = monthly.dropna()

        # ML forecast for total sales
        X = monthly.drop(['DATE','SALES AMT'], axis=1)
        y = monthly['SALES AMT']
        model = XGBRegressor(n_estimators=100, max_depth=3)
        model.fit(X[:-1], y[:-1])
        next_pred = model.predict(X[-1:])[0]

        st.subheader("Total Sales - Historical & Forecast")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Latest Actual Month", f"{y.iloc[-1]:,.0f}")
        with c2:
            st.metric("Next Month Forecast", f"{next_pred:,.0f}")

        st.line_chart(monthly.set_index('DATE')['SALES AMT'])

        # Product-level forecast
        st.subheader("Product-wise Sales Prediction")
        products = df['PRODUCT NAME'].unique()
        prod_result = []
        for prod in products:
            dfp = df[df['PRODUCT NAME'] == prod].copy()
            if len(dfp) < 7:
                continue
            mprod = dfp.groupby(dfp['DATE'].dt.to_period('M'))['SALES AMT'].sum().reset_index()
            mprod['DATE'] = mprod['DATE'].dt.to_timestamp()
            mprod['month'] = mprod['DATE'].dt.month
            mprod['lag1'] = mprod['SALES AMT'].shift(1)
            mprod['rolling3'] = mprod['SALES AMT'].shift(1).rolling(3).mean()
            mprod = mprod.dropna()
            if len(mprod) < 2:
                continue
            Xp = mprod.drop(['DATE','SALES AMT'], axis=1)
            yp = mprod['SALES AMT']
            modelp = XGBRegressor(n_estimators=50, max_depth=3)
            modelp.fit(Xp[:-1], yp[:-1])
            next_predp = modelp.predict(Xp[-1:])[0]
            prod_result.append({'Product': prod, 'Forecast_Next_Month': next_predp, 'Last_Actual_Month': yp.iloc[-1]})

        if prod_result:
            prod_df = pd.DataFrame(prod_result).sort_values('Forecast_Next_Month', ascending=False)
            st.dataframe(prod_df.style.format({'Forecast_Next_Month': '{:,.0f}', 'Last_Actual_Month': '{:,.0f}'}), height=500)
        else:
            st.warning("Not enough history for product-wise forecast.")

        # Optionally, group by territory
        '''
        if 'TERRITORY' in df.columns:
            st.subheader("Territory-wise Sales (Current Month)")
            territory_sales = df.groupby('TERRITORY')['SALES AMT'].sum().sort_values(ascending=False)
            st.bar_chart(territory_sales)'''

        st.success("You can upload new data at the start of every month and see updated predictions!")
else:
    st.info("Please upload your Excel sales data file to begin.")

st.caption("Built with ❤️ using Streamlit and XGBoost.")

