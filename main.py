from keys import *  # Make sure this includes: API_KEY, SECRET_KEY, and maybe client setup
import pandas as pd
import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create the Alpaca client here if not imported from keys.py
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Day,
    start=datetime.datetime(2020, 1, 1),
    end=datetime.datetime.now()
)

bars = client.get_stock_bars(request).df
dataset = bars[bars.index.get_level_values(0) == 'SPY'].copy()
dataset.reset_index(inplace=True, drop=True)

def plot_stock_data(df):
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.03,
                         subplot_titles=('Candlestick with SMAs', 'Volume'),
                         row_width=[0.3, 0.7])

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'],
                             line=dict(color='orange', width=1),
                             name='SMA 5'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'],
                             line=dict(color='blue', width=1),
                             name='SMA 20'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'],
                             line=dict(color='purple', width=1),
                             name='SMA 50'),
                  row=1, col=1)

    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'],
                         marker_color=colors,
                         name='Volume'),
                  row=2, col=1)

    fig.update_layout(
        title='SPY Stock Price and Volume Analysis',
        yaxis_title='Stock Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=600
    )

    fig.show()

def add_indicators(df):
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    df['Price_Change'] = df['close'].pct_change()

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # Momentum (rate of change)
    df['Momentum'] = df['close'].diff(4)

    # Lag feature
    df['Close_Lag1'] = df['close'].shift(1)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Binary target: will price go up tomorrow?
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

    return df


def prepare_data(df, look_back=10):
    features = ['close', 'SMA5', 'SMA20', 'SMA50', 'Price_Change',
                'MACD', 'Momentum', 'RSI', 'Close_Lag1', 'volume']
    
    df = df.dropna()
    print("Target distribution:\n", df['Target'].value_counts(normalize=True))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    x, y = [], []
    for i in range(look_back, len(scaled_data) - 1):
        x.append(scaled_data[i-look_back:i])
        y.append(df['Target'].iloc[i])

    return np.array(x), np.array(y)


def create_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    symbol = 'SPY'
    look_back = 10

    print("Fetching stock data...")
    df = dataset.copy()

    print("Plotting stock data...")
    plot_stock_data(df)

    print("Calculating technical indicators...")
    df = add_indicators(df)

    print("Preparing data for LSTM...")
    X, y = prepare_data(df, look_back)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("Training model...")
    model = create_model(input_shape=[look_back, X.shape[2]])


    # Calculate class weights for imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    class_weight=class_weight_dict
    )

    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)

    print("\nModel Performance Explanation:")
    print(f"Train accuracy: {train_score[1]:.4f}")
    print(f"Test accuracy: {test_score[1]:.4f}")

    last_sequence = X[-1:]
    tomorrow_pred = model.predict(last_sequence)[0][0]

    print(f"\nPrediction for Tomorrow:")
    print(f"Probability of price increase: {tomorrow_pred:.2%}")
    print(f"Predicted direction: {'UP' if tomorrow_pred > 0.5 else 'DOWN'}")
