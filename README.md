# Algorithmic binary options trading
Test your algorithmic binary options trading strategy on financial market data from Yahoo Finance.

> :warning: _This must be intended for research purposes only and must not be used for real trading applications!_

## Requirements
The `binary-options.py` script requires Python 3.8 and a bunch of modules that can be installed by typing `pip install -r requirements.txt` in your terminal.

## How does it work?
It automatically retrieve the last day financial market data about a specific ticker organised in one-minute data time points from Yahoo Finance. It currently combines the simple RSI and the Stochastic Oscillator in order to determine whether it could be the right time to trade.
