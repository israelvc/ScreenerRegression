from libraries import Input, Output, go, make_subplots
from libraries import pd, yf
from kernelRegression import calculate_indicator

def registerMainPlotting(app):
    @app.callback(
        Output('ohlc-plot', 'figure'),
        Input('stocksInList', 'value'),
        Input('rsiLength', 'value'),
        Input('stochasticLength', 'value'),
        Input('cciLength', 'value')
    )
    def showOHLCAndIndicators(selectedTicker, rsi_length, stochastic_length, cci_length):
        if not selectedTicker:
            return go.Figure()
        
        data = yf.download(selectedTicker,
                           period="2y",
                           interval="1d")
        
        if data.empty:
            return go.Figure()

        # print("checking NAs ", data.isna().sum())

        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(key=selectedTicker, axis=1, level='Ticker')

        # Ensure datetime index and numeric OHLC columns
        data.index = pd.to_datetime(data.index)
        data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].astype(float)

        lookback = 60
        tuning = 15.0
        variant = "Tuneable"

        # Calculate signal
        data['cosineKernelSignal'] = calculate_indicator(data['Close'],
                                             data['High'],
                                             data['Low'],
                                             data['Close'],
                                             lookback,
                                             tuning,
                                             variant,
                                             rsiLength = rsi_length,
                                             stochasticLength = stochastic_length,
                                             cciLength = cci_length)

        print("Added cosine kernel signal")

        # ================
        # Create subplots
        # ================
        ohlc_indicators_fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=["OHLC Chart", "Cosine kernel"]#, "EMA volume"]
        )
        # Adding OHLC plot
        ohlc_indicators_fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        # Add cosine kernel signal
        ohlc_indicators_fig.add_trace(
            go.Scatter(
                x = data.index,
                y = data['cosineKernelSignal'],
                mode='lines',
                name='Cosine Kernel signal'
            ),
            row = 2, col=1
        )

        return ohlc_indicators_fig.to_dict()