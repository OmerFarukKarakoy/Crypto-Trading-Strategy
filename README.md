# Crypto-Trading-Strategy

## Overview

The **Crypto Trading Strategy 3 Bot** is an AI-powered trading system that uses **RSI**, **EMA**, and **WaveTrend** indicators to identify market trends and generate buy/sell signals. It also includes an **LSTM-based model** to forecast future prices and offers **Turkish-language recommendations** with visual analysis support.



## Features

- **Technical Indicator Analysis**: Calculates RSI, EMA, and WaveTrend oscillator for real-time market assessment.
- **Signal Generation**: Applies Strategy 3 rules to issue Strong Buy, Reactive Buy, Sell, Strong Sell, or Hold signals.
- **LSTM Price Forecasting**: Uses deep learning (LSTM) to predict future price movement.
- **Backtesting System**: Simulates past trades to evaluate strategy performance.
- **Turkish Recommendations**: Outputs personalized Turkish-language trade suggestions with explanation and confidence level.
- **Visualization**: Generates multi-panel charts including EMA, RSI, WaveTrend, and volume analysis.

## Technologies Used

- **Python**: Core programming language for all development.
- **Pandas & NumPy**: For data preprocessing and computation.
- **Matplotlib**: For detailed multi-panel financial visualization.
- **TensorFlow / Keras**: For building and training the LSTM model.
- **Binance API**: To fetch real-time and historical crypto data.
- **yFinance** *(optional)*: Backup source for testing with sample data.

## Usage

1. Clone the repo:
   git clone ...

2. Install requirements:
   pip install -r requirements.txt

3. Set up `.env`:
   API_KEY=...
   API_SECRET=...

4. Run:
   python Kripto2.7.py

5. Enter symbol (e.g. BTCUSDT) and review the analysis

## 🖥️ Preview 
### 🔍 Cryptocurrency Selection Panel
![Ekran görüntüsü](https://github.com/user-attachments/assets/82ad2ca1-9b2a-4b76-bb18-f5b1fd7170d1)

![Ekran görüntüsü2](https://github.com/user-attachments/assets/42fd6d27-eb5c-42e0-b1ac-0bb3eaf8eab7)

### 🤖 Strategy Analysis Results
![Ekran görüntüsü3](https://github.com/user-attachments/assets/8e89f57e-a62b-4c74-977d-ff48ef888875)

![Ekran görüntüsü4](https://github.com/user-attachments/assets/4dfb4c2e-7747-4230-8748-18a2f07a725f)

### 📊 Technical Analysis View  
![Figure_1](https://github.com/user-attachments/assets/23735f1f-7d62-4183-ad35-d77bfae1558b)





## License

This project is licensed under the MIT License.  
© 2025 Ömer Faruk Karakoy — You are free to use, modify, and distribute this software.  
Provided "as is", without warranty of any kind.
