#!/usr/bin/env python3
"""
Test binary option trading strategies on financial market data
"""

__author__ = "Fabio Cumbo (fabio.cumbo@gmail.com)"
__version__ = "0.1.0"
__date__ = "Jan 6, 2023"

import argparse as ap
import errno
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf

from plotly.subplots import make_subplots
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.momentum import StochasticOscillator


def read_params():
    p = ap.ArgumentParser(
        description="A simple dashboard for playing with financial market data",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        help="Initial balance in USD",
    )
    p.add_argument(
        "--bet",
        type=float,
        default=50.0,
        help="Fixed bet in USD",
    )
    p.add_argument(
        "--data",
        type=os.path.abspath,
        help="Path to the file with input data. This will avoid retrieving data from Yahoo Finance",
    )
    p.add_argument(
        "--double-when-loss",
        action="store_true",
        default=False,
        dest="double_when_loss",
        help="Double bet in case of loss",
    )
    p.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Organise retrieved Yahoo Finance data in intervals",
    )
    p.add_argument(
        "--limit-loss",
        type=float,
        default=100.0,
        dest="limit_loss",
        help="Stop playing in case the total loss is greater than this amount in USD",
    )
    p.add_argument(
        "--limit-profit",
        type=float,
        default=200.0,
        dest="limit_profit",
        help="Stop playing in case the total profit is greater than this amount in USD",
    )
    p.add_argument(
        "--period",
        type=str,
        default="1d",
        help="Retrieve data back in time from Yahoo Finance up to this time period",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Plot market data and indicators",
    )
    p.add_argument(
        "--profit",
        type=float,
        default=90.0,
        help="Profit percentage on the bet amount",
    )
    p.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker",
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version="version {} ({})".format(__version__, __date__),
        help="Print the tool version and exit",
    )
    return p.parse_args()


def autoplay(
    data: pd.DataFrame, 
    balance: float=1000.0, 
    bet: float=50.0, 
    profit: float=90.0,
    loss_limit: float=100.0,
    profit_limit: float=200.0,
    double_when_loss: bool=False
) -> None:
    """
    Autoplay algorithmic binary option trading on financial market data based on RSI and Stochastic Oscillator

    :param data:                Pandas DataFrame with financial market data
    :param balance:             Initial balance in USD
    :param bet:                 Fixed bet in USD
    :param profit:              Profit percentage on the bet amount
    :param loss_limit:          Stop playing in case the total loss is greater than this amount in USD
    :param profit_limit:        Stop playing in case the total profit is greater than this amount in USD
    :param double_when_loss:    Double bet in case of loss
    """

    # Take track of the initial balance and bet
    initial_balance = balance
    initial_bet = bet
    
    print("Balance: {}\n".format(balance))

    # Compute the Stichastic Oscillator
    stoch = StochasticOscillator(high=data["High"],
                                 close=data["Close"],
                                 low=data["Low"],
                                 window=14,
                                 smooth_window=3)
    
    # Compute the RSI Indicator
    rsi = RSIIndicator(close=data["Close"],
                       window=14)
    
    # Get the number of data points
    data_points = len(data.index)

    # Define bet params
    bet_check = False
    bet_time = 2 # minutes
    bet_open_time = None
    bet_open_value = None
    bet_up = False
    bet_down = False

    # Counter on data time points
    data_count = 1

    # Iterate over data time points
    for timep, openv, sst, ssi, rv in zip(data.index.tolist(),           # Market data time points
                                          data["Open"].tolist(),         # Open values
                                          stoch.stoch().tolist(),        # Stochastic Oscillator
                                          stoch.stoch_signal().tolist(), # Stochastic signal
                                          rsi.rsi().tolist()):           # RSI indicator
        
        if bet_check:
            # Bet
            bet_time = 2
            bet_open_time = timep
            bet_open_value = openv
            bet_check = False
            print("Bet ({}): {}\t{}".format("Up" if bet_up else "Down", timep, openv))
            balance -= bet
        
        # Check whether you can bet
        if bet_open_time == None:
            # There should be enough data time points
            if data_points-data_count > bet_time:
                # There should be enough balance
                if balance-bet > 0.0 and initial_balance+profit_limit > balance and balance > initial_balance-loss_limit:
                    if sst > 80.0 and ssi > 80.0 and sst > ssi and rv > 80.0:
                        # Wait a minute and bet
                        bet_check = True
                        bet_down = True
                    elif sst < 20.0 and ssi < 20.0 and sst < ssi and rv < 20.0:
                        # Wait a minute and bet
                        bet_check = True
                        bet_up = True
        
        if bet_open_value != None:
            # Time to check if the strategy worked
            if bet_time == 0:
                if bet_down and bet_open_value > openv:
                    print("Win: {}\t{}".format(timep, openv))
                    # Update balance
                    balance += bet + (bet*profit/100.0)
                    bet = initial_bet
                elif bet_up and bet_open_value < openv:
                    print("Win: {}\t{}".format(timep, openv))
                    # Update balance
                    balance += bet + (bet*profit/100.0)
                    bet = initial_bet
                else:
                    print("Lose: {}\t{}".format(timep, openv))
                    # Double bet in case of loss (optional)
                    if double_when_loss:
                        bet *= 2

                # Reset bet params
                print("Balance: {}\n".format(balance))
                bet_time = 2
                bet_open_time = None
                bet_open_value = None
                bet_down = False
                bet_up = False
            
            else:
                # Check if the strategy works when the bet time reaches 0
                bet_time -= 1
        
        # Increment counter on data time points
        data_count += 1


def download(ticker: str, period: str="1d", interval: str="1m") -> pd.DataFrame:
    """
    Download marked data from Yahoo Finance

    :param ticker:      Ticker name
    :param period:      Retrieve data back in time up to the specified period
    :param interval:    Organise data according to this time interval
    :return:            Pandas DataFrame with market data
    """

    # Ask Yahoo Finance
    return yf.download(tickers=ticker, period=period, interval=interval)


def load(filepath: str) -> pd.DataFrame:
    """
    Load data from file into a Pandas DataFrame

    :param filepath:    Path to the input file with financial market data
    :return:            Pandas DataFrame with market data
    """

    return pd.read_csv(filepath, sep=",", header=0, index_col=0)


def plot(data: pd.DataFrame, ticker: str) -> None:
    """
    Plot financial market data with Plotly

    :param data:    Pandas DataFrame with market data
    :param ticker:  Ticker name
    """

    # Add Moving Averages (5-day and 20-day)
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA20"] = data["Close"].rolling(window=20).mean()

    # MACD
    macd = MACD(close=data["Close"],
                window_slow=26,
                window_fast=12,
                window_sign=9)
    
    # Stichastic Oscillator
    stoch = StochasticOscillator(high=data["High"],
                                 close=data["Close"],
                                 low=data["Low"],
                                 window=14,
                                 smooth_window=3)

    # Declare figure
    fig = go.Figure()

    # Add subplot properties
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01,
                        row_heights=[0.5, 0.1, 0.2, 0.2])

    # Define candles
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Market data"
    ))
    
    # Add 5-day Moving Average trace
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["MA5"],
        opacity=0.7,
        line=dict(color="blue", width=2),
        name="MA 5"
    ))

    # Add 20-day Moving Average trace
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["MA20"],
        opacity=0.7,
        line=dict(color="orange", width=2),
        name="MA 20"
    ))

    # Plot volume trace on 2nd row
    colors = ["green" if row["Open"] - row["Close"] >= 0 else "red" for _, row in data.iterrows()]
    
    fig.add_trace(
        go.Bar(x=data.index,
               y=data["Volume"],
               marker_color=colors
        ),
        row=2,
        col=1
    )

    # Plot MACD trace on 3rd row
    colors = ["green" if value >= 0 else "red" for value in macd.macd_diff()]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=macd.macd_diff(),
            marker_color=colors
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=macd.macd(),
            line=dict(color="black", width=2)
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=macd.macd_signal(),
            line=dict(color="blue", width=1)
        ),
        row=3,
        col=1
    )

    # Plot stochastics trace on 4th row
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=stoch.stoch(),
            line=dict(color="black", width=2)
        ),
        row=4,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=stoch.stoch_signal(),
            line=dict(color="blue", width=1)
        ),
        row=4,
        col=1
    )

    # Update layout by changing the plot size, hiding legends and rangeslider, and removing gaps between dates
    fig.update_layout(height=900, width=1200,
                      showlegend=False,
                      xaxis_rangeslider_visible=False)

    # Add titles
    fig.update_layout(title="Price evolution ({})".format(ticker),
                      yaxis_title="Price (USD)")

    # Y-Axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", showgrid=False, row=3, col=1)
    fig.update_yaxes(title_text="Stoch", row=4, col=1)

    # X-Axes
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # Show
    fig.show()


def main() -> None:
    # Load command line parameters
    args = read_params()

    if args.data:
        if not os.path.isfile(args.data):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.data)
        
        # Load market data from file
        data = load(args.data)

        # Check whether the input dataset contains columns "Open", "Close", "High", "Low", and "Volume"
        required = {"Open", "Close", "High", "Low", "Volume"}
        missing = required.intersection(set(data.columns))
        if missing:
            raise Exception("Missing columns: [{}]".format(",".join(missing)))

    else:
        # Download market data from Yahoo Finance
        data = download(args.ticker, period=args.period, interval=args.interval)

    if args.plot:
        # Plot market data with Plotly
        plot(data, args.ticker)

    # Play!
    autoplay(
        data, 
        balance=args.balance, 
        bet=args.bet, 
        profit=args.profit, 
        loss_limit=args.limit_loss, 
        profit_limit=args.limit_profit, 
        double_when_loss=args.double_when_loss
    )


if __name__ == "__main__":
    main()
