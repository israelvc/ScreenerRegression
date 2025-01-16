from libraries import Input, Output, State
from libraries import pd, yf
from pathlib import Path
from kernelRegression import calculate_indicator

def registerStocksInSelectedScreenFile(app):
     @app.callback(
          Output('stocksInList', 'options'),
          Output('stocksInList', 'value'),
          Input('screenFileList', 'value')
     )
     def extractTickersInSelectedFile(selectedScreenFile):
          data_folder = Path("data")
          selected_file = data_folder / f"{selectedScreenFile}"
          screenFileData = pd.read_csv(selected_file)
          tickersAvailable = screenFileData['Symbol'].sort_values()

          return tickersAvailable, tickersAvailable[0]

def reigsterFilterScreenedTickers(app):
    @app.callback(
        Output('filteredSecurities', 'options'),
        Input('filterScreenedSecurities', 'n_clicks'),
        State('screenFileList', 'value')
    )
    def filter_tickers(n_clicks, screenFile):
        if not n_clicks:
            # Prevent update if button hasn't been clicked
            return []
        
        print("Initiating screening process")
        filtered_tickers = []
        failed_tickers = []
        
        data_folder = Path("data")
        selected_file = data_folder / f"{screenFile}"
        screenFileData = pd.read_csv(selected_file)
        
        for ticker in screenFileData['Symbol']:
            print("Analyzing ",ticker)
            tickerData = yf.download(ticker,
                                     period="6mo",
                                     interval="1d")
            
            if tickerData.empty:
                print("Failed to load data from ticker, adding it to failed list")
                failed_tickers.append(ticker)
                continue
            
            if isinstance(tickerData.columns, pd.MultiIndex):
                    tickerData = tickerData.xs(key=ticker, axis=1, level='Ticker')

            # Ensure datetime index and numeric OHLC columns
            tickerData.index = pd.to_datetime(tickerData.index)
            tickerData[['Open', 'High', 'Low', 'Close']] = tickerData[['Open', 'High', 'Low', 'Close']].astype(float)

            ########### Add cosine kernel regression py file and call it to add signal as column
            lookback = 144
            tuning = 15.0
            variant = "Tuneable"

            # Calculate signal
            signal = calculate_indicator(tickerData['Close'],
                                         tickerData['High'],
                                         tickerData['Low'],
                                         tickerData['Close'],
                                         lookback,
                                         tuning,
                                         variant,
                                         rsiLength = 5,
                                         stochasticLength=5,
                                         cciLength=13)
            print(signal)
        
        return filtered_tickers