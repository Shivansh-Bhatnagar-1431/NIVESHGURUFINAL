import streamlit as st
import nselib
from nselib import capital_market
from nselib import derivatives
import pandas as pd
from datetime import datetime, timedelta

# Streamlit App Configuration
st.set_page_config(page_title="NSE Market Dashboard", layout="wide")

# Title
st.title("NSE Market Dashboard")

# Sidebar for User Input
st.sidebar.header("Data Selection")
data_type = st.sidebar.selectbox("Select Data Type", ["Price Volume", "Bhav Copy", "Future Price"])
symbol = st.sidebar.text_input("Enter Symbol (e.g., SBIN, BANKNIFTY)", "SBIN")

# Custom date range input, limited to past and present dates
today = datetime.now().date()
from_date = st.sidebar.date_input("From Date", min_value=today - timedelta(days=365), max_value=today, value=today - timedelta(days=30))
to_date = st.sidebar.date_input("To Date", min_value=today - timedelta(days=365), max_value=today, value=today)

# Function to Fetch Data
def fetch_data(data_type, symbol, from_date, to_date):
    from_date_str = from_date.strftime('%d-%m-%Y')
    to_date_str = to_date.strftime('%d-%m-%Y')

    if data_type == "Price Volume":
        data = capital_market.price_volume_and_deliverable_position_data(
            symbol=symbol, 
            from_date=from_date_str, 
            to_date=to_date_str
        )
    elif data_type == "Bhav Copy":
        data = capital_market.bhav_copy_with_delivery(trade_date=to_date_str)
    elif data_type == "Future Price":
        data = derivatives.future_price_volume_data(
            symbol=symbol, 
            instrument='FUTSTK', 
            from_date=from_date_str, 
            to_date=to_date_str
        )
    return data

# Main Dashboard
if st.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        try:
            data = fetch_data(data_type, symbol, from_date, to_date)
            
            # Display Data
            st.subheader(f"{data_type} Data for {symbol}")
            st.dataframe(data)
            
            # Basic Visualization (if applicable)
            if 'DATE' in data.columns and 'CLOSE_PRICE' in data.columns:
                data['DATE'] = pd.to_datetime(data['DATE'])
                st.line_chart(data.set_index('DATE')['CLOSE_PRICE'])
            elif 'TRADE_DATE' in data.columns and 'CLOSE' in data.columns:
                data['TRADE_DATE'] = pd.to_datetime(data['TRADE_DATE'])
                st.line_chart(data.set_index('TRADE_DATE')['CLOSE'])

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Trading Holiday Calendar Button
if "show_holidays" not in st.session_state:
    st.session_state.show_holidays = False

if st.sidebar.button("Show Trading Holiday Calendar"):
    st.session_state.show_holidays = not st.session_state.show_holidays

if st.session_state.show_holidays:
    st.subheader("Trading Holiday Calendar")
    holiday_data = nselib.trading_holiday_calendar()
    st.dataframe(holiday_data)
    if st.button("Close Calendar"):
        st.session_state.show_holidays = False

# Footer
st.sidebar.markdown("Powered by NiveshGuru")
