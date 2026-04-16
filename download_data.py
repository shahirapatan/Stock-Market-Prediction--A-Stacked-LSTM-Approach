import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Get today's date
end_date = datetime.today()
start_date = end_date - timedelta(days=2*365)

# Company symbols
stocks = {
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "HDFCBank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN"
}

# Create Excel file
with pd.ExcelWriter("Indian_International_Stock_Data.xlsx") as writer:
    
    for name, symbol in stocks.items():
        data = yf.download(symbol, start=start_date, end=end_date)
        data.to_excel(writer, sheet_name=name)

print("All company data downloaded successfully!")