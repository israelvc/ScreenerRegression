from libraries import Dash, dcc, html, daq
import initializationFile
import Callbacks.sidebarCallbacks as sidebarCallbacks
import Callbacks.ohlcPlottingCallback as ohlcPlottingCallback

screenedFiles = initializationFile.loadScreenFiles()

app = Dash(__name__)
app.title = "Data Dashboard"

sidebarCallbacks.reigsterFilterScreenedTickers(app)
sidebarCallbacks.registerStocksInSelectedScreenFile(app)
ohlcPlottingCallback.registerMainPlotting(app)

app.layout = html.Div(
    children=[
        # Sidebar
        html.Div(
            children=[
                html.H2("Securities Screener",
                        style={'text-align': 'center'}),
                html.Label("Select a coin:"),
                dcc.Dropdown(
                    id = 'screenFileList',
                    options = screenedFiles,
                    value = screenedFiles[0],
                    placeholder="Select a security",
                    style={'margin-bottom': '10px'}
                ),
                dcc.Dropdown(
                    id = 'stocksInList',
                    # options = screenedFiles,
                    # value = screenedFiles[0],
                    placeholder="Select a security",
                    style={'margin-bottom': '10px'}
                ),
                html.H2("Cosine kernel settings",
                        style={'text-align': 'center'}),
                html.Label("RSI length:"),
                daq.NumericInput(
                    id='rsiLength',
                    value=5,
                    size = 70,
                    style={'margin-bottom': '10px'}
                ),
                html.Label("Stochastic length:"),
                daq.NumericInput(
                    id='stochasticLength',
                    value=5,
                    size = 70,
                    style={'margin-bottom': '10px'}
                ),
                html.Label("CCI length:"),
                daq.NumericInput(
                    id='cciLength',
                    value = 13,
                    size = 70,
                    style={'margin-bottom': '10px'}
                ),
                dcc.Dropdown(
                    id = 'screeningAlgorithm',
                    options = [
                        {'label': 'Cosine kernel', 'value': 'CosineKernel'},
                        {'label': 'Tripple kernel', 'value': 'TrippleKernel'}
                    ],
                    value='CosineKernel',
                    style={'margin-bottom': '10px'}
                ),
                html.Button(
                    'Screen securities',
                    id = 'filterScreenedSecurities',
                    n_clicks = 0,
                    style={
                        #'margin-top': 'auto',  # Push the button to the bottom
                        'align-self': 'center',  # Center the button horizontally
                        'padding': '10px 20px',
                        'backgroundColor': '#007BFF',
                        'color': '#FFF',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'fontSize': '16px'
                    }
                ),
                dcc.Dropdown(
                    id = 'filteredSecurities',
                    # options = screenedFiles,
                    # value='GRT',
                    placeholder="Select a security",
                    style={'margin-top': '10px'}
                )
            ],
            style={
                'width': '20%',
                'backgroundColor': '#f0f0f0',
                'padding': '10px',
                'boxShadow': '2px 0 5px rgba(0, 0, 0, 0.1)',
                'height': '100%',  # Make the sidebar stretch vertically
                'display': 'flex',
                'flexDirection': 'column'
            }
        ),
        # Main content (plot)
        html.Div(
            children=[
                dcc.Graph(
                    id='ohlc-plot',
                    style={'height': '100%', 'width': '100%'}
                )
            ],
            style={
                'flex': '1',
                'padding': '10px',
                'height': '100%'  # Stretch the main content vertically
            }
        )
    ],
    style={
        'display': 'flex',  # Align sidebar and plot side by side
        'height': '100vh',  # Set the container to the full viewport height
        'margin': '0'
    }
)

if __name__ == "__main__":
    app.run_server(debug=True)
