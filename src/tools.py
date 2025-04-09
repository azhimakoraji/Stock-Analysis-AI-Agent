import datetime as dt 
from typing import Union, Dict, Set, List, TypedDict, Annotated
import pandas as pd
from langchain_core.tools import tool
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicator for a given ticker."""
    try:
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=24*3),
            end=dt.datetime.now(),
            interval='1wk'
        )
        df= data.copy()
        data.reset_index(inplace=True)
        data.Date = data.Date.astype(str)
        
        indicators = {}
        
        rsi_series = RSIIndicator(df['Close'].squeeze(), window=14).rsi().iloc[-12:]
        indicators["RSI"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in rsi_series.dropna().to_dict().items()}
        
        sto_series = StochasticOscillator(
            df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze(), window=14).stoch().iloc[-12:]
        indicators["Stochastic_Oscillator"] = {
                    date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in sto_series.dropna().to_dict().items()}

        macd = MACD(df['Close'].squeeze())
        macd_series = macd.macd().iloc[-12:]
        indicators["MACD"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in macd_series.to_dict().items()}
        
        macd_signal_series = macd.macd_signal().iloc[-12:]
        indicators["MACD_Signal"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in macd_signal_series.to_dict().items()}
        
        vwap_series = volume_weighted_average_price(
            high=df['High'].squeeze(), low=df['Low'].squeeze(), close=df['Close'].squeeze(), 
            volume=df['Volume'].squeeze(),
        ).iloc[-12:]
        indicators["vwap"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in vwap_series.to_dict().items()}
        
        return {'stock_price': data.to_dict(orient='records'),
                'indicators': indicators}

    except Exception as e:
        return f"Error fetching price data: {str(e)}"
    
@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if info is missing or invalid
        if not info or not isinstance(info, dict):
            return "Financial metrics data is unavailable or incomplete."

        # Extract financial metrics safely with default values
        return {
            'pe_ratio': info.get('forwardPE', 'N/A'),
            'price_to_book': info.get('priceToBook', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'profit_margins': info.get('profitMargins', 'N/A'),
            'return_on_equity': info.get('returnOnEquity', 'N/A'),
            'dividend_per_share': info.get('dividendRate', 'N/A'),
            'market_growth': info.get('trailingEpsGrowth', 'N/A'),
            'net_margin': info.get('netMargins', 'N/A'),
            'current_ratio': info.get('currentRatio', 'N/A'),
            'quick_ratio': info.get('quickRatio', 'N/A'),
            'free_cash_flow': info.get('freeCashflow', 'N/A'),
            'revenue_growth': info.get('revenueGrowth', 'N/A')
        }

    except Exception as e:
        return f"Error fetching financial metrics: {str(e)}"

@tool
def analyze_stock_sentiment(ticker: str) -> Union[Dict, str]:
    """Evaluates the sentiment of news articles associated with the stock using Yahoo Finance and VaderSentiment."""
    try:
        # Retrieve stock-related news using yfinance
        stock_data = yf.Ticker(ticker)
        related_news = stock_data.news  # Collect stock news articles

        if not related_news:
            return f"No articles available for ticker {ticker}."

        # Gather news headlines
        headlines = [entry["title"] for entry in related_news]

        # Perform sentiment analysis via VaderSentiment
        sentiment_analyzer = SentimentIntensityAnalyzer()
        headline_sentiments = {
            headline: sentiment_analyzer.polarity_scores(headline) for headline in headlines
        }

        # Compute average sentiment across all headlines
        sentiment_averages = {
            "positive": sum(item["pos"] for item in headline_sentiments.values()) / len(headline_sentiments),
            "neutral": sum(item["neu"] for item in headline_sentiments.values()) / len(headline_sentiments),
            "negative": sum(item["neg"] for item in headline_sentiments.values()) / len(headline_sentiments),
            "compound": sum(item["compound"] for item in headline_sentiments.values()) / len(headline_sentiments),
        }

        # Assign overall sentiment
        general_sentiment = "Positive" if sentiment_averages["compound"] > 0 else "Negative"

        return {
            'ticker_symbol': ticker,
            'headlines': headlines,
            'detailed_sentiment': headline_sentiments,
            'average_sentiment_scores': sentiment_averages,
            'overall_sentiment': general_sentiment
        }

    except Exception as e:
        return f"Error occurred during sentiment analysis: {str(e)}"



@tool
def calculate_volatility(ticker: str) -> Union[Dict, str]:
    """Analyzes the stock's historical volatility over a given period."""
    try:
        # Download historical stock data
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=52),  # 1 year of weekly data
            end=dt.datetime.now(),
            interval='1d'
        )

        if data.empty or 'Close' not in data:
            return "Insufficient data to calculate volatility."

        # Calculate daily returns
        data['Daily_Return'] = data['Close'].pct_change()

        # Compute volatility metrics
        volatility_metrics = {
            'standard_deviation': round(data['Daily_Return'].std() * (252**0.5), 4),  # Annualized volatility
            'average_daily_return': round(data['Daily_Return'].mean(), 4),
            'max_drawdown': round((data['Close'] / data['Close'].cummax() - 1).min(), 4),
        }

        return {
            'ticker_symbol': ticker,
            'volatility_metrics': volatility_metrics,
            'time_period': '1 year'
        }

    except Exception as e:
        return f"Error occurred while calculating volatility: {str(e)}"



@tool
def forecast_stock_price(ticker: str) -> Union[Dict, str]:
    """Forecasts future stock prices using a time series model."""
    try:
        # Fetch historical stock data
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=104),  # 2 years of weekly data
            end=dt.datetime.now(),
            interval='1wk'
        )

        # Ensure data is available and valid
        if data.empty or 'Close' not in data.columns or len(data) < 52:
            return "Insufficient data to forecast stock prices. At least 1 year of weekly data is required."

        # Process the data for the model
        df = data[['Close']].dropna()
        if df.empty or len(df) < 52:
            return "Insufficient data after processing. Forecasting requires at least 1 year of weekly data."

        # Convert data to float type
        df['Close'] = df['Close'].astype(float)

        # Fit the Exponential Smoothing model
        model = ExponentialSmoothing(df['Close'], seasonal='add', seasonal_periods=52)
        try:
            fitted_model = model.fit()
        except Exception as model_error:
            return f"Error fitting the model: {str(model_error)}"

        # Forecast the next 12 weeks
        forecast = fitted_model.forecast(12)

        return {
            'forecast': {f"Week {i+1}": round(price, 2) for i, price in enumerate(forecast)},
            'model_summary': str(fitted_model.summary())
        }

    except Exception as e:
        return f"Error forecasting stock prices: {str(e)}"



@tool
def compare_with_peers(ticker: str, peers: List[str]) -> Union[Dict, str]:
    """Compares a stock's financial metrics with its peers."""
    try:
        # Fetch financial metrics for the main ticker
        stock = yf.Ticker(ticker)
        info = stock.info

        # Ensure valid data is available for the main ticker
        if not info or not isinstance(info, dict):
            return f"Unable to retrieve financial data for {ticker}."

        main_metrics = {
            'pe_ratio': info.get('forwardPE', 'N/A'),
            'price_to_book': info.get('priceToBook', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'profit_margins': info.get('profitMargins', 'N/A')
        }

        # Fetch financial metrics for peers
        peer_metrics = {}
        for peer in peers:
            peer_stock = yf.Ticker(peer)
            peer_info = peer_stock.info
            if not peer_info or not isinstance(peer_info, dict):
                peer_metrics[peer] = "No data available"
            else:
                peer_metrics[peer] = {
                    'pe_ratio': peer_info.get('forwardPE', 'N/A'),
                    'price_to_book': peer_info.get('priceToBook', 'N/A'),
                    'debt_to_equity': peer_info.get('debtToEquity', 'N/A'),
                    'profit_margins': peer_info.get('profitMargins', 'N/A')
                }

        return {
            'main_ticker': ticker,
            'main_metrics': main_metrics,
            'peer_comparison': peer_metrics
        }

    except Exception as e:
        return f"Error occurred while comparing with peers: {str(e)}"


