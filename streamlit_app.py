import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.markdown("""
    <style>
    .big-font {font-size:30px !important;}
    .center {text-align: center;}
    .blue-box {background-color: #E3F2FD; padding: 20px; border-radius: 15px;}
    .subsection {font-size:22px; color:#1565C0; font-weight:bold;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font center"><b>📈 Sales Forecasting Dashboard</b></p>', unsafe_allow_html=True)

st.markdown("""
<div class='center blue-box'>
Upload your <b>Sales Data Excel file</b> (same format as your dataset).<br>
See sales trends and <b>predict next month</b> by Product and Territory!
</div>
""", unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader("Upload sales data (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("**First 5 rows:**")
    st.dataframe(df.head())

    if 'DATE' not in df.columns or 'SALES AMT' not in df.columns:
        st.error("File must have columns 'DATE' and 'SALES AMT'")
        st.stop()
    df['DATE'] = pd.to_datetime(df['DATE'])

    st.markdown('<div class="subsection">Overall Monthly Sales Trend & Forecast</div>', unsafe_allow_html=True)
    monthly = df.groupby(df['DATE'].dt.to_period('M'))['SALES AMT'].sum().reset_index()
    monthly['DATE'] = monthly['DATE'].dt.to_timestamp()
    monthly['month'] = monthly['DATE'].dt.month
    monthly['lag1'] = monthly['SALES AMT'].shift(1)
    monthly['rolling3'] = monthly['SALES AMT'].shift(1).rolling(3).mean()
    monthly = monthly.dropna()
    X = monthly.drop(['DATE','SALES AMT'], axis=1)
    y = monthly['SALES AMT']
    model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X[:-1], y[:-1])
    overall_forecast = model.predict(X[-1:])[0]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Latest Actual Month", f"{y.iloc[-1]:,.0f}")
    with c2:
        st.metric("Next Month Forecast", f"{overall_forecast:,.0f}")

    st.line_chart(monthly.set_index('DATE')['SALES AMT'])

    # ---- PRODUCT-WISE ----
    st.markdown('<div class="subsection">Product-wise Forecast (Next Month)</div>', unsafe_allow_html=True)
    with st.expander("Show/Hide Product-wise Forecast Table"):
        prod_result = []
        if 'PRODUCT NAME' in df.columns:
            products = df['PRODUCT NAME'].unique()
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
                modelp = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
                modelp.fit(Xp[:-1], yp[:-1])
                predp = modelp.predict(Xp[-1:])[0]
                prod_result.append({'Product': prod, 'Forecast_Next_Month': predp, 'Last_Actual_Month': yp.iloc[-1]})
            if prod_result:
                prod_df = pd.DataFrame(prod_result).sort_values('Forecast_Next_Month', ascending=False)
                st.dataframe(prod_df.style.format({'Forecast_Next_Month': '{:,.0f}', 'Last_Actual_Month': '{:,.0f}'}), height=500)
            else:
                st.info("Not enough history for product-wise forecast.")
        else:
            st.info("No 'PRODUCT NAME' column in the file.")

    # ---- TERRITORY-WISE ----
    st.markdown('<div class="subsection">Territory-wise Forecast (Next Month)</div>', unsafe_allow_html=True)
    with st.expander("Show/Hide Territory-wise Forecast Table"):
        terr_result = []
        if 'TERRITORY' in df.columns:
            territories = df['TERRITORY'].unique()
            for terr in territories:
                dft = df[df['TERRITORY'] == terr].copy()
                if len(dft) < 7:
                    continue
                mterr = dft.groupby(dft['DATE'].dt.to_period('M'))['SALES AMT'].sum().reset_index()
                mterr['DATE'] = mterr['DATE'].dt.to_timestamp()
                mterr['month'] = mterr['DATE'].dt.month
                mterr['lag1'] = mterr['SALES AMT'].shift(1)
                mterr['rolling3'] = mterr['SALES AMT'].shift(1).rolling(3).mean()
                mterr = mterr.dropna()
                if len(mterr) < 2:
                    continue
                Xt = mterr.drop(['DATE','SALES AMT'], axis=1)
                yt = mterr['SALES AMT']
                modelt = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
                modelt.fit(Xt[:-1], yt[:-1])
                predt = modelt.predict(Xt[-1:])[0]
                terr_result.append({'Territory': terr, 'Forecast_Next_Month': predt, 'Last_Actual_Month': yt.iloc[-1]})
            if terr_result:
                terr_df = pd.DataFrame(terr_result).sort_values('Forecast_Next_Month', ascending=False)
                st.dataframe(terr_df.style.format({'Forecast_Next_Month': '{:,.0f}', 'Last_Actual_Month': '{:,.0f}'}), height=500)
            else:
                st.info("Not enough history for territory-wise forecast.")
        else:
            st.info("No 'TERRITORY' column in the file.")

    # --- Territory Actuals Chart ---
    if 'TERRITORY' in df.columns:
        st.markdown('<div class="subsection">Territory-wise Current Month Actual Sales</div>', unsafe_allow_html=True)
        cur_month = df['DATE'].dt.to_period('M').max()
        latest = df[df['DATE'].dt.to_period('M') == cur_month]
        territory_sales = latest.groupby('TERRITORY')['SALES AMT'].sum().sort_values(ascending=False)
        st.bar_chart(territory_sales)

    # --- Download Option (Product & Territory) ---
    st.markdown('<div class="subsection">Download Forecasts</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    if 'prod_df' in locals():
        c3.download_button("Download Product Forecasts", prod_df.to_csv(index=False), "product_forecast.csv", "text/csv")
    if 'terr_df' in locals():
        c4.download_button("Download Territory Forecasts", terr_df.to_csv(index=False), "territory_forecast.csv", "text/csv")

    st.success("Upload new data each month for updated forecasts. Built for easy business use!")
else:
    st.info("Please upload your Excel sales data to begin.")

st.caption("Built with ❤️ by Nethma | Powered by Streamlit & XGBoost")
