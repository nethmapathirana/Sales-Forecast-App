import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("🧮 Sales Forecasting Dashboard")

st.write(
    """
    Sales data is loaded from a file on Google Drive.
    The app will show trends and predict sales for the next month using Machine Learning (XGBoost).
    """
)

# -- CONFIGURE GOOGLE DRIVE FILE --
# 1. Upload Excel to Google Drive
# 2. Right-click > "Share" > "Anyone with the link"
# 3. Copy the share link, for example:
#    https://drive.google.com/file/d/FILE_ID/view?usp=sharing
# 4. Replace FILE_ID below:

#GOOGLE_DRIVE_FILE_ID = "1h10O11yLxIVxPwVEN2DDexoqlF-9Lpml"
download_url = "https://drive.google.com/uc?id=1A2UR7sWb9WJKgC-FC7nd6ZUKG2vUt8rr"

@st.cache_data(show_spinner=True)
def load_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

df = load_data(download_url)
if df is not None:
    if 'DATE' not in df.columns or 'SALES AMT' not in df.columns:
        st.error("Excel must have 'DATE' and 'SALES AMT' columns.")
    else:
        df['DATE'] = pd.to_datetime(df['DATE'])
        st.write("First 5 rows:", df.head())
        st.info(f"Rows: {len(df):,}, Columns: {df.columns.tolist()}")

        # --- TOTAL SALES FORECAST ---
        monthly = df.groupby(df['DATE'].dt.to_period('M'))['SALES AMT'].sum().reset_index()
        monthly['DATE'] = monthly['DATE'].dt.to_timestamp()
        monthly['month'] = monthly['DATE'].dt.month
        monthly['lag1'] = monthly['SALES AMT'].shift(1)
        monthly['rolling3'] = monthly['SALES AMT'].shift(1).rolling(3).mean()
        monthly = monthly.dropna()

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

        # --- PRODUCT-WISE FORECAST ---
        st.subheader("Product-wise Sales Prediction")
        if 'PRODUCT NAME' in df.columns:
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

        # --- TERRITORY-WISE FORECAST ---
        st.subheader("Territory-wise Sales Prediction")
        if 'TERRITORY' in df.columns:
            territories = df['TERRITORY'].unique()
            terr_result = []
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
                modelt = XGBRegressor(n_estimators=50, max_depth=3)
                modelt.fit(Xt[:-1], yt[:-1])
                next_predt = modelt.predict(Xt[-1:])[0]
                terr_result.append({'Territory': terr, 'Forecast_Next_Month': next_predt, 'Last_Actual_Month': yt.iloc[-1]})

            if terr_result:
                terr_df = pd.DataFrame(terr_result).sort_values('Forecast_Next_Month', ascending=False)
                st.dataframe(terr_df.style.format({'Forecast_Next_Month': '{:,.0f}', 'Last_Actual_Month': '{:,.0f}'}), height=500)
            else:
                st.warning("Not enough history for territory-wise forecast.")
        else:
            st.info("No 'TERRITORY' column in the data.")

        st.success("Update the file in Google Drive each month and the dashboard will update automatically.")
else:
    st.info("Loading sales data from Google Drive...")

st.caption("Built with ❤️ using Streamlit and XGBoost.")

