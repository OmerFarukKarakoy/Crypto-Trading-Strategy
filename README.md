# Crypto-Trading-Strategy

## Overview

The **Crypto Trading Strategy 3 Bot** is an AI-powered trading system that uses **RSI**, **EMA**, and **WaveTrend** indicators to identify market trends and generate buy/sell signals. It also includes an **LSTM-based model** to forecast future prices and offers **Turkish-language recommendations** with visual analysis support.
The bot provides recommendations based on a four-tier signal system:

🟢 "Güçlü Al" → Strong Buy: All indicators (RSI, EMA, WaveTrend) confirm a bullish signal.

🟦 "Temkinli Al" → Cautious Buy: Core indicators (EMA and WaveTrend) align positively, but RSI does not confirm.

⚪ "Bekle" → Wait: EMA and WaveTrend give conflicting signals (e.g., one BUY, one SELL). RSI is ignored.

🟧 "Temkinli Sat" → Cautious Sell: Core indicators (EMA and WaveTrend) align negatively, but RSI does not confirm.

🔴 "Güçlü Sat" → Strong Sell: All indicators (RSI, EMA, WaveTrend) confirm a bearish signal.

This structure ensures clarity and precision in trading decisions, allowing users to quickly interpret the strength and direction of market signals in real time.



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

![Ekran görüntüsü 2](https://github.com/user-attachments/assets/21fbdab3-d403-4fc3-87ae-20241dd152d7)

### 🤖 Strategy Analysis Results
![Ekran görüntüsü 3](https://github.com/user-attachments/assets/b8d46f70-e666-4256-a4d9-8ac7afa02f9d)

![Ekran görüntüsü 4](https://github.com/user-attachments/assets/d7b18c2d-c4ec-4488-a366-d63a1e1db5b1)


### 📊 Technical Analysis View  
![Figure_1](https://github.com/user-attachments/assets/b594ef83-9ca7-430d-950f-6d31ccb48abb)

## ⚠️ Disclaimer

🚨 **Educational Purposes Only**  
• Not investment advice  
• Contains theoretical strategies  
• Use at your own risk  

*"Crypto trading involves substantial risk of loss."*


## License

This project is licensed under the MIT License.  
© 2025 Ömer Faruk Karakoy — You are free to use, modify, and distribute this software.  
Provided "as is", without warranty of any kind.
