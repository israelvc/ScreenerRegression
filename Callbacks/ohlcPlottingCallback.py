from libraries import Input, Output, go, make_subplots
from libraries import pd, yf
from kernelRegression import prepareDataForCosineKernelRegression, calculate_indicator

def registerMainPlotting(app):
    @app.callback(
        Output('ohlc-plot', 'figure'),
        Input('stocksInList', 'value'),
        Input('rsiLength', 'value'),
        Input('stochasticLength', 'value'),
        Input('cciLength', 'value'),
        Input('cmoLength', 'value'),
        Input('bbpctLength', 'value'),
        Input('fisherTransformLength', 'value'),
        Input('vzoLength', 'value')
    )
    def showOHLCAndIndicators(selectedTicker,
                              rsi_length, stochastic_length, cci_length,
                              cmo_length, bbpct_length, fisher_transform_length, vzo_length):
        if not selectedTicker:
            return go.Figure()
        
        data = yf.download(selectedTicker,
                           period="2y",
                           interval="1d")
        
        if data.empty:
            return go.Figure()

        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(key=selectedTicker, axis=1, level='Ticker')

        # Ensure datetime index and numeric OHLC columns
        data.index = pd.to_datetime(data.index)
        data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].astype(float)

        # Prepare data
        data = prepareDataForCosineKernelRegression(
            tickerData = data,
            rsiLength = rsi_length,
            stochasticLength = stochastic_length,
            cciLength = cci_length,
            cmoLength = cmo_length,
            bbpctLength = bbpct_length,
            fisherTransformLength = fisher_transform_length,
            vzoLength = vzo_length
        )

        # Calculate cosine kernel regression
        data = calculate_indicator(
            dataSource = data,
            lookbackR = 144,
            tuning = 15,
            variant = "Tuneable"
        )

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
                y = data['out'],
                mode='lines',
                name='Cosine Kernel signal'
            ),
            row = 2, col=1
        )
        ohlc_indicators_fig.add_trace(
            go.Scatter(
                x = data.index,
                y = data['out2'],
                mode='lines',
                name='Cosine Kernel signal'
            ),
            row = 2, col=1
        )

        # Update layout to reduce range slider height and configure appearance
        ohlc_indicators_fig.update_layout(
            autosize=True,
            height=None,
            title="OHLC and Cosine Kernel Signals",
            xaxis=dict(
                rangeslider=dict(
                    visible=True,
                    thickness=0.03
                )
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False
        )

        return ohlc_indicators_fig.to_dict()