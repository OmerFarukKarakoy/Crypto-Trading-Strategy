# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 19:57:18 2025

@author: omerf
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 03:34:22 2025

@author: omerf
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 19:16:08 2025

@author: omerf
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import logging
import os
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

class TechnicalIndicators:
    """Class to calculate technical indicators without talib"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI without talib"""
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    @staticmethod
    def calculate_ema(prices, period=11):
        """Calculate Exponential Moving Average without talib"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def calculate_sma(prices, period):
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values
    
    @staticmethod
    def calculate_wavetrend(high, low, close, channel_length=10, average_length=21):
        """Calculate WaveTrend Oscillator without talib"""
        # Typical Price
        hlc3 = (high + low + close) / 3
        hlc3_series = pd.Series(hlc3)
        
        # Calculate EMA of typical price
        esa = hlc3_series.ewm(span=channel_length, adjust=False).mean()
        
        # Calculate absolute difference
        abs_diff = np.abs(hlc3_series - esa)
        d = abs_diff.ewm(span=channel_length, adjust=False).mean()
        
        # Calculate CI (Commodity Channel Index component)
        ci = (hlc3_series - esa) / (0.015 * d)
        
        # Calculate WaveTrend
        wt1 = ci.ewm(span=average_length, adjust=False).mean()
        wt2 = wt1.rolling(window=4).mean()
        
        return wt1.values, wt2.values
    
    @staticmethod
    def calculate_wavetrend(high, low, close, channel_length=10, average_length=21):
        """WaveTrend hesaplama (aşırı bölgeler dahil)"""
        hlc3 = (high + low + close) / 3
        esa = pd.Series(hlc3).ewm(span=channel_length, adjust=False).mean()
        d = pd.Series(np.abs(hlc3 - esa)).ewm(span=channel_length, adjust=False).mean()
        ci = (hlc3 - esa) / (0.015 * d)
        wt1 = pd.Series(ci).ewm(span=average_length, adjust=False).mean()
        wt2 = wt1.rolling(window=4).mean()
        
        # Aşırı alım/satım bölgeleri (±60)
        wt_overbought = 60    # Standart değer
        wt_oversold = -60     # Standart değer
        
        return wt1.values, wt2.values, wt_overbought, wt_oversold

class LSTMModel:
    """LSTM Model for price prediction"""
    
    def __init__(self, lookback_period=60):
        self.lookback_period = lookback_period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def prepare_data(self, data):
        """Prepare data for LSTM training"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.lookback_period, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_period:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train(self, data, epochs=50, batch_size=32):
        """Train the LSTM model"""
        X, y = self.prepare_data(data)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build and train model
        self.model = self.build_model((X_train.shape[1], 1))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return history
    
    def predict(self, data):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Prepare data for prediction
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        if len(scaled_data) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} data points for prediction")
        
        # Get last sequence
        last_sequence = scaled_data[-self.lookback_period:].reshape((1, self.lookback_period, 1))
        
        # Make prediction
        prediction = self.model.predict(last_sequence, verbose=0)
        
        # Inverse transform
        return self.scaler.inverse_transform(prediction)[0][0]

class Strategy3TradingBot:
    """Strategy 3 Trading Bot - Optimized for 1H timeframe"""
    
    def __init__(self, symbol, period="1h", lookback_days=100):
        self.symbol = symbol
        self.period = period
        self.lookback_days = lookback_days
        self.data = None
        self.lstm_model = LSTMModel()
        self.indicators = TechnicalIndicators()
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.entry_price = 0
        self.trade_history = []
        self.wavetrend_buy_signal_active = False  # Track if WT buy signal is active
        
    def fetch_data(self):
        """Fetch data from Binance API"""
        from binance.client import Client
        import pandas as pd

        # Binance API anahtarları
        API_KEY = 'cfDC92B191b9B3Ca3D842Ae0e01108CBKI6BqEW6xr4NrPus3hoZ9Ze9YrmWwPFV'
        API_SECRET = 'f9AbA6a8AD6bC2a97294a212244dda04ETfl0kc4BSUGOtL7m7rNELpt3Jh25SiP'
        client = Client(API_KEY, API_SECRET)

        try:
            print(f"📡 Binance üzerinden veri çekiliyor: {self.symbol}")
            symbol_binance = self.symbol.replace("-", "")  # BTC-USD → BTCUSDT
            
            klines = client.get_klines(symbol=self.symbol, interval='1h', limit=500)

            df = pd.DataFrame(klines, columns=[
                'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
            ])

            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul')  # İstanbul saati için
            df.set_index('Timestamp', inplace=True)
            df = df.astype(float)
            self.data = df

            print(self.data[['Open', 'High', 'Low', 'Close']].head())
            return True

        except Exception as e:
            print(f"❌ Binance veri çekme hatası: {e}")
            return False
    
    def create_sample_data(self):
        """Create sample data for testing when API fails"""
        print("📊 Örnek veri oluşturuluyor...")
        
        # Generate sample price data
        np.random.seed(42)
        current_date = datetime.now().strftime("%Y-%m-%d")
        dates = pd.date_range(start=current_date, periods=500, freq='H')
        
        # Create realistic price movement
        initial_price = 100
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create OHLC data
        high_factor = np.random.uniform(1.001, 1.02, len(dates))
        low_factor = np.random.uniform(0.98, 0.999, len(dates))
        
        self.data = pd.DataFrame({
            'Open': prices,
            'High': np.array(prices) * high_factor,
            'Low': np.array(prices) * low_factor,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        print(f"✅ {len(self.data)} örnek veri noktası oluşturuldu")
        return True
    
    def calculate_indicators(self):
        """Calculate all technical indicators according to Strategy 3"""
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Fetch data first.")
        
        # Strategy 3 specific settings
        # RSI: Period = 14, levels: 35 (oversold), 50 (neutral), 70 (overbought)
        self.data['RSI'] = self.indicators.calculate_rsi(self.data['Close'].values, period=14)
        
        # EMA: Period = 11 (optimal for RSI 14)
        self.data['EMA'] = self.indicators.calculate_ema(self.data['Close'].values, period=11)
        
        # WaveTrend'i hesapla (aşırı bölgeleri de döndür)
        wt1, wt2, wt_overbought, wt_oversold = self.indicators.calculate_wavetrend(
        self.data['High'].values,
        self.data['Low'].values,
        self.data['Close'].values
        )
   
        self.data['WT1'] = wt1
        self.data['WT2'] = wt2
        self.data['WT_Overbought'] = wt_overbought  # +60
        self.data['WT_Oversold'] = wt_oversold      # -60
   
        # Alım/Satım sinyallerini aşırı bölgelerle güncelle
        self.data['WT_Buy_Signal'] = False
        self.data['WT_Sell_Signal'] = False
        self.data['WT_Zero_Cross_Up'] = False  # Sıfır çizgisini yukarı geçiş
        self.data['WT_Zero_Cross_Down'] = False  # Sıfır çizgisini aşağı geçiş
    
        for i in range(1, len(self.data)):
            # 1. Tüm WT1-WT2 kesişimleri (aşırı bölge şartı yok)
            if (self.data['WT1'].iloc[i] > self.data['WT2'].iloc[i] and 
                self.data['WT1'].iloc[i-1] <= self.data['WT2'].iloc[i-1]):
                self.data.loc[self.data.index[i], 'WT_Buy_Signal'] = True
            
            if (self.data['WT1'].iloc[i] < self.data['WT2'].iloc[i] and 
                self.data['WT1'].iloc[i-1] >= self.data['WT2'].iloc[i-1]):
                self.data.loc[self.data.index[i], 'WT_Sell_Signal'] = True
            
            # 2. Sıfır çizgisi geçişleri
            if (self.data['WT1'].iloc[i] > 0 and 
                self.data['WT1'].iloc[i-1] <= 0):
                self.data.loc[self.data.index[i], 'WT_Zero_Cross_Up'] = True
            
            if (self.data['WT1'].iloc[i] < 0 and 
                self.data['WT1'].iloc[i-1] >= 0):
                self.data.loc[self.data.index[i], 'WT_Zero_Cross_Down'] = True
        
        # Add price vs EMA relationship
        self.data['Price_Above_EMA'] = self.data['Close'] > self.data['EMA']
        self.data['Price_Cross_Above_EMA'] = False
        
        # Detect price crossing above EMA
        for i in range(1, len(self.data)):
            if (self.data['Close'].iloc[i] > self.data['EMA'].iloc[i] and 
                self.data['Close'].iloc[i-1] <= self.data['EMA'].iloc[i-1]):
                self.data.loc[self.data.index[i], 'Price_Cross_Above_EMA'] = True
    
    def train_lstm(self):
        """Train LSTM model on historical data"""
        if self.data is None:
            raise ValueError("No data available. Fetch data first.")
        
        print("🧠 LSTM modeli eğitiliyor...")
        close_prices = self.data['Close'].values
        
        try:
            history = self.lstm_model.train(close_prices, epochs=50, batch_size=32)
            print("✅ LSTM model başarıyla eğitildi!")
            return history
        except Exception as e:
            print(f"❌ LSTM eğitim hatası: {e}")
            return None
    
    def generate_signal_strategy3(self, index):
        """
    Generate trading signal based on updated Strategy 3 requirements:
        
        BUY SIGNALS:
        - Signal 2 (Güçlü Al): RSI ≤ 35 + WT buy signal + Price crosses above EMA
        - Signal 1 (Al Tepkimli): WT buy signal + Price crosses above EMA (RSI ignored)
        
        SELL SIGNALS:
        - Signal -2 (Güçlü Sat): RSI ≥ 70 + WT sell signal + Price below EMA
        - Signal -1 (Sat): Any single sell condition met
        
        NEUTRAL:
        - Signal 0 (Nötr): None of the above conditions
        """
        if index < 60:  # Need enough data for indicators
            return 0
    
        current_data = self.data.iloc[index]
    
        # Get current values
        rsi = current_data['RSI']
        close_price = current_data['Close']
        ema = current_data['EMA']
        wt1 = current_data['WT1']
        wt2 = current_data['WT2']
        wt_oversold = current_data['WT_Oversold']  # Typically -60
        wt_overbought = current_data['WT_Overbought']  # Typically +60
    
        # Check recent WT buy signals (last 5 periods)
        wt_buy_recent = False
        for i in range(max(0, index-5), index+1):
            if (self.data['WT1'].iloc[i] > self.data['WT2'].iloc[i] and 
                self.data['WT1'].iloc[i-1] <= self.data['WT2'].iloc[i-1]):
                """and self.data['WT1'].iloc[i] < wt_oversold):"""
                wt_buy_recent = True
                break
    
        # Check current WT sell signal
        wt_sell_signal = (wt1 < wt2 and 
                          wt1 > wt_overbought and 
                          wt2 > wt_overbought and
                          self.data['WT1'].iloc[index-1] >= self.data['WT2'].iloc[index-1])
    
        # Price conditions
        price_cross_above_ema = current_data['Price_Cross_Above_EMA']
        price_above_ema = current_data['Price_Above_EMA']
    
        # Individual conditions
        rsi_oversold = rsi <= 35
        rsi_overbought = rsi >= 70
    
        # BUY CONDITIONS
        # Güçlü Al: All 3 conditions (RSI ≤ 35 + WT buy + Price cross above EMA)
        strong_buy = rsi_oversold and wt_buy_recent and price_cross_above_ema
        
        # Al (Tepkimli): WT buy + Price cross above EMA (RSI ignored)
        reactive_buy = wt_buy_recent and price_cross_above_ema and not strong_buy
    
        # SELL CONDITIONS
        # Güçlü Sat: All 3 sell conditions
        strong_sell = rsi_overbought and wt_sell_signal and not price_above_ema
        
        # Regular Sat: Any single sell condition
        regular_sell = (rsi_overbought or wt_sell_signal or not price_above_ema) and not strong_sell
    
        # Generate signals
        if strong_buy:
            return 2  # Güçlü Al
        elif reactive_buy:
            return 1  # Al (Tepkimli)
        elif strong_sell:
            return -2  # Güçlü Sat
        elif regular_sell:
            return -1  # Sat
        else:
            return 0  # Nötr (Bekle)
    
    def execute_trade(self, signal, price, timestamp):
        """Execute trade based on signal"""
        if signal == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price
            self.trade_history.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': price,
                'position': self.position
            })
            print(f"🟢 BUY at ${price:.2f} on {timestamp}")
            
        elif signal == -1 and self.position == 1:  # Sell
            profit = price - self.entry_price
            profit_pct = (profit / self.entry_price) * 100
            
            self.position = 0
            self.trade_history.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': price,
                'profit': profit,
                'profit_pct': profit_pct,
                'position': self.position
            })
            print(f"🔴 SELL at ${price:.2f} on {timestamp}, Profit: ${profit:.2f} ({profit_pct:.2f}%)")
    
    def backtest(self):
        """Run backtest on historical data using Strategy 3"""
        if self.data is None:
            raise ValueError("No data available. Fetch data first.")
        
        print("⏮️ Strategy 3 backtest başlatılıyor...")
        
        for i in range(60, len(self.data)):
            signal = self.generate_signal_strategy3(i)
            price = self.data['Close'].iloc[i]
            timestamp = self.data.index[i].strftime("%d.%m.%Y %H:%M")
            
            self.execute_trade(signal, price, timestamp)
        
        # Close any open position
        if self.position == 1:
            last_price = self.data['Close'].iloc[-1]
            last_timestamp = self.data.index[-1]
            self.execute_trade(-1, last_price, last_timestamp)
        
        return self.analyze_performance()
    
    def analyze_performance(self):
        """Analyze trading performance"""
        if not self.trade_history:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Calculate metrics
        total_trades = len(trades_df[trades_df['action'] == 'SELL'])
        
        if total_trades == 0:
            return {"error": "No completed trades"}
        
        profits = trades_df[trades_df['action'] == 'SELL']['profit'].values
        profit_pcts = trades_df[trades_df['action'] == 'SELL']['profit_pct'].values
        
        total_profit = profits.sum()
        avg_profit = profits.mean()
        win_rate = len(profits[profits > 0]) / len(profits) * 100
        max_profit = profits.max()
        max_loss = profits.min()
        
        performance = {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'win_rate': win_rate,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'avg_profit_pct': profit_pcts.mean(),
            'total_profit_pct': profit_pcts.sum()
        }
        
        return performance
    
    def get_current_signal_strategy3(self):
        """Get current trading signal based on Strategy 3"""
        if self.data is None or len(self.data) < 60:
            return None
        
        # Get LSTM prediction
        try:
            predicted_price = self.lstm_model.predict(self.data['Close'].values)
            current_price = self.data['Close'].iloc[-1]
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
        except:
            predicted_price = None
            price_change_pct = None
        
        # Get Strategy 3 signal
        technical_signal = self.generate_signal_strategy3(len(self.data) - 1)
        
        current_data = self.data.iloc[-1]
        
        # Check Strategy 3 conditions status
        rsi_condition = current_data['RSI'] <= 35
        wt_recent = any(self.data['WT_Buy_Signal'].iloc[max(0, len(self.data)-6):])
        price_above_ema = current_data['Price_Above_EMA']
        current_time = datetime.now().strftime("%d.%m.%Y %H:%M")
        
        return {
            'timestamp': current_time,
            'current_price': current_data['Close'],
            'predicted_price': predicted_price,
            'price_change_prediction': price_change_pct,
            'rsi': current_data['RSI'],
            'rsi_oversold': rsi_condition,
            'ema': current_data['EMA'],
            'price_above_ema': price_above_ema,
            'wt1': current_data['WT1'],
            'wt2': current_data['WT2'],
            'wt_buy_signal': current_data['WT_Buy_Signal'],
            'wt_recent_signal': wt_recent,
            'strategy3_signal': technical_signal,
            'signal_description': self.get_strategy3_signal_description(technical_signal, rsi_condition, wt_recent, price_above_ema, rsi_value=None)
        }
    
    def get_strategy3_signal_description(self, signal, rsi_oversold, wt_recent, price_above_ema, rsi_value=None):
        """Get human-readable Strategy 3 signal description with enhanced emojis and explanations"""
        if signal == 2:  # Güçlü Al - all 3 conditions met
            return "🟢 GÜÇLÜ AL - Tüm Strateji 3 koşulları sağlandı: 📉 RSI aşırı satım (≤35), 🌊 WaveTrend alım sinyali aktif ✅, 📈 Fiyat EMA üzerinde çapraz yaptı ✅"
        elif signal == 1:  # Al (Tepkimli) - WT + EMA cross, RSI ignored
            rsi_status = f"📊 RSI: {rsi_value:.1f}" if rsi_value is not None else ""
            return f"🟢 AL (TEPKIMLİ) - WaveTrend + EMA koşulları sağlandı: 🌊 WaveTrend alım sinyali aktif ✅, 📈 Fiyat EMA üzerinde çapraz yaptı ✅ ({rsi_status})"
        elif signal == -2:  # Güçlü Sat
            return "🔴 GÜÇLÜ SAT - Çoklu çıkış sinyali: 📈 RSI aşırı alım (≥70) VE 🌊 WaveTrend satım sinyali VE 📉 Fiyat EMA altına düştü"
        elif signal == -1:  # Sat
            return "🔴 SAT - Çıkış koşulları sağlandı: 📈 RSI aşırı alım (≥70) VEYA 🌊 WaveTrend satım sinyali VEYA 📉 Fiyat EMA altına düştü"
        else:  # Nötr (Bekle)
            # Provide detailed status of each condition
            conditions = []
            if rsi_oversold:
                conditions.append("✅ RSI aşırı satım bölgesinde 📉")
            else:
                    rsi_status = f"📊 RSI: {rsi_value:.1f}" if rsi_value is not None else "normal seviyede"
                    conditions.append(f"❌ RSI {rsi_status}")
            
            if wt_recent:
                conditions.append("✅ WaveTrend alım sinyali aktif 🌊")
            else:
                    conditions.append("❌ WaveTrend alım sinyali yok 🌊")
            
            if price_above_ema:
                conditions.append("✅ Fiyat EMA üzerinde 📈")
            else:
                conditions.append("❌ Fiyat EMA altında 📉")
        
            return f"🔵 NÖTR (BEKLE) - Strateji 3 koşul durumu: {' | '.join(conditions)}"
    
    def get_turkish_recommendation_strategy3(self):
        """Updated Turkish trading recommendation with new labeling system"""
        if self.data is None or len(self.data) < 60:
            return {
                'action': 'BEKLE',
                'status': 'Yetersiz veri',
                'explanation': 'Analiz için yeterli veri bulunmuyor.',
                'confidence': 'Düşük',
                'risk_level': None
            }
        
        # Para birimi sembolünü belirle (BTCUSDT -> USDT, BTCTRY -> TRY)
        quote_currency = self.symbol[-3:] if len(self.symbol) >= 3 else self.symbol[-4:]
        
        # Özel para birimi formatları
        currency_formats = {
            'TRY': lambda x: f"{x:,.2f}₺",       # 1,234.56₺
            'USDT': lambda x: f"${x:,.2f}",      # $1,234.56
            'SDT': lambda x: f"${x:,.2f}",
            'BTC': lambda x: f"₿{x:.8f}".rstrip('0').rstrip('.'),  # ₿0.00123456
            'ETH': lambda x: f"Ξ{x:.6f}".rstrip('0').rstrip('.'),  # Ξ0.123456
            'BNB': lambda x: f"{x:.4f} BNB"      # 1.2345 BNB
        }
        format_price = currency_formats.get(quote_currency, lambda x: f"{x:.2f} {quote_currency}")
        
        current_data = self.data.iloc[-1]
        technical_signal = self.generate_signal_strategy3(len(self.data) - 1)
        
        rsi = current_data['RSI']
        ema = current_data['EMA']
        close = current_data['Close']
        wt1 = current_data['WT1']
        wt2 = current_data['WT2']
        
        # Updated conditions
        rsi_oversold = rsi <= 35
        rsi_overbought = rsi >= 70
        wt_recent = any(self.data['WT_Buy_Signal'].iloc[max(0, len(self.data)-6):])
        price_above_ema = close > ema
        price_cross_above_ema = current_data['Price_Cross_Above_EMA']
        
        # LSTM tahmini
        try:
            predicted_price = self.lstm_model.predict(self.data['Close'].values)
            price_change_pct = ((predicted_price - close) / close) * 100
            lstm_bullish = price_change_pct > 1
            lstm_bearish = price_change_pct < -1
        except Exception as e:
            print(f"LSTM tahmin hatası: {e}")
            lstm_bullish = False
            lstm_bearish = False
            price_change_pct = 0
        
        # Updated signal handling with new labeling system
        if technical_signal == 2:  # Güçlü Al
            action = "🟢 GÜÇLÜ AL - TÜM KOŞULLAR SAĞLANDI"
            status = "Mükemmel Giriş Fırsatı"
            explanation = (f"📉 RSI aşırı satım ({rsi:.1f} ≤ 35), "
                          f"🌊 WaveTrend alım sinyali aktif, "
                          f"💰 Fiyat ({format_price(close)}) EMA üzerinde çapraz yaptı ({format_price(ema)})")
            confidence = "Çok Yüksek"
            risk_level = "Çok Düşük"
            
        elif technical_signal == 1:  # Al (Tepkimli)
            action = "🟢 AL (TEPKIMLİ) - GÜÇLÜ MOMENTUM"
            status = "WaveTrend + EMA Sinyali"
            explanation = (f"🌊 WaveTrend alım sinyali aktif, "
                          f"💰 Fiyat ({format_price(close)}) EMA üzerinde çapraz yaptı ({format_price(ema)}), "
                          f"📊 RSI: {rsi:.1f} (göz ardı edildi)")
            confidence = "Yüksek"
            risk_level = "Düşük"
            
        elif technical_signal == -2:  # Güçlü Sat
            action = "🔴 GÜÇLÜ SAT - ÇOKLU ÇIKIŞ SİNYALİ"
            status = "Acil Çıkış Gerekli"
            explanation = (f"📈 RSI aşırı alım ({rsi:.1f} ≥ 70), "
                          f"🌊 WaveTrend satım sinyali, "
                          f"📉 Fiyat ({format_price(close)}) EMA altında ({format_price(ema)})")
            confidence = "Çok Yüksek"
            risk_level = "Çok Düşük"
            
        elif technical_signal == -1:  # Sat
            action = "🔴 SAT - ÇIKIŞ SİNYALİ"
            status = "Satış Zamanı"
            explanation_parts = []
            if rsi_overbought:
                explanation_parts.append(f"📈 RSI aşırı alım ({rsi:.1f} ≥ 70)")
            if not price_above_ema:
                explanation_parts.append(f"📉 Fiyat ({format_price(close)}) EMA altında ({format_price(ema)})")
            if wt1 < wt2:
                explanation_parts.append("🌊 WaveTrend sat sinyali")
                
            explanation = "⚠️ Çıkış sinyali: " + ", ".join(explanation_parts) + "."
            confidence = "Yüksek"
            risk_level = "Orta"
            
        else:  # Nötr (Bekle)
            action = "🔵 NÖTR (BEKLE)"
            status = "Bekleme Modunda"
            
            # Check how many conditions are met for detailed feedback
            conditions_met = sum([rsi_oversold, wt_recent, price_cross_above_ema])
            
            if conditions_met == 2:
                status = "2/3 Koşul Sağlandı - Yakın Takip"
                missing = []
                if not rsi_oversold: 
                    missing.append(f"📊 RSI > 35 ({rsi:.1f})")
                if not wt_recent: 
                    missing.append("🌊 WaveTrend sinyali yok")
                if not price_cross_above_ema: 
                    missing.append("📉 Fiyat EMA çaprazı yok")
                explanation = f"⏳ Eksik koşul: {', '.join(missing)}"
                confidence = "Orta"
                risk_level = "Orta"
                
            elif conditions_met == 1:
                status = "1/3 Koşul Sağlandı - Sabırlı Bekle"
                explanation = "⚠️ Güçlü sinyal için daha fazla koşul gerekiyor."
                confidence = "Düşük"
                risk_level = "Yüksek"
                
            else:
                status = "Koşullar Sağlanmadı"
                explanation = f"📊 RSI: {rsi:.1f}, 💰 Fiyat: {format_price(close)}, 📈 EMA: {format_price(ema)}"
                confidence = "Düşük"
                risk_level = "Yüksek"
        
        # Add LSTM insights
        if lstm_bullish and technical_signal >= 0:
            explanation += f" 🤖 LSTM %{price_change_pct:.1f} artış öngörüyor - pozitif destek."
        elif lstm_bearish and technical_signal <= 0:
            explanation += f" 🤖 LSTM %{abs(price_change_pct):.1f} düşüş öngörüyor - negatif baskı."
        elif abs(price_change_pct) > 2:
            explanation += f" 🤖 LSTM %{price_change_pct:.1f} değişim öngörüyor."
        
        return {
            'action': action,
            'status': status,
            'explanation': explanation,
            'confidence': confidence,
            'risk_level': risk_level,
            'current_price': format_price(close),
            'rsi': f"{rsi:.1f}",
            'ema': format_price(ema),
            'wt1': f"{wt1:.2f}",
            'wt2': f"{wt2:.2f}",
            'lstm_prediction': f"%{price_change_pct:.1f}" if abs(price_change_pct) > 0 else "N/A",
            'signal_code': technical_signal  # For debugging/tracking
        }
        
    
    def plot_analysis_strategy3(self, save_path="strategy3_analysis.png"):
        """Geliştirilmiş EMA ve WaveTrend sinyalleri ile analiz grafiği"""
        if self.data is None or len(self.data) < 60:
            print("⚠️ Grafik için yeterli veri yok")
            return

        plt.rcParams['font.family'] = ['Segoe UI', 'Arial', 'Calibri', 'sans-serif']
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        fig.suptitle(f'🎯 Geliştirilmiş Strateji 3 Analizi - {self.symbol} (1 Saatlik)', 
                    fontsize=16, fontweight='bold')

        # 1. Fiyat ve EMA (Tüm kesişim sinyalleri ile)
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], 
                label='💰 Fiyat', linewidth=2, color='black')
        ax1.plot(self.data.index, self.data['EMA'], 
                label='📈 EMA (11)', linewidth=2, color='orange', alpha=0.8)

        # Yukarı yönlü EMA kesişimleri (Fiyat EMA'nın üstüne çıkarsa)
        cross_above = []
        for i in range(1, len(self.data)):
            if (self.data['Close'].iloc[i] > self.data['EMA'].iloc[i] and 
                self.data['Close'].iloc[i-1] <= self.data['EMA'].iloc[i-1]):
                cross_above.append(self.data.index[i])

        # Aşağı yönlü EMA kesişimleri (Fiyat EMA'nın altına inerse)
        cross_below = []
        for i in range(1, len(self.data)):
            if (self.data['Close'].iloc[i] < self.data['EMA'].iloc[i] and 
                self.data['Close'].iloc[i-1] >= self.data['EMA'].iloc[i-1]):
                cross_below.append(self.data.index[i])

        # Kesişim noktalarını çiz
        if cross_above:
            ax1.scatter(cross_above, 
                       [self.data.loc[date, 'Close'] for date in cross_above],
                       color='green', marker='^', s=100, 
                       label='🟢 Fiyat EMA Üstüne Çıktı', zorder=5)
        
        if cross_below:
            ax1.scatter(cross_below, 
                       [self.data.loc[date, 'Close'] for date in cross_below],
                       color='red', marker='v', s=100, 
                       label='🔴 Fiyat EMA Altına Düştü', zorder=5)

        ax1.set_title('💹 Fiyat ve EMA Analizi - Tüm Kesişimler', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. RSI (Aynı kalacak)
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], 
                label='📊 RSI (14)', linewidth=2, color='purple')
        ax2.axhline(y=70, color='red', linestyle='--', 
                    alpha=0.7, label='📈 Aşırı Alım (70)')
        ax2.axhline(y=50, color='gray', linestyle='-', 
                    alpha=0.5, label='⚖️ Nötr (50)')
        ax2.axhline(y=35, color='green', linestyle='--', 
                    alpha=0.7, label='📉 Aşırı Satım (35)')

        oversold_zones = self.data[self.data['RSI'] <= 35]
        if not oversold_zones.empty:
            ax2.fill_between(oversold_zones.index, 0, oversold_zones['RSI'],
                            color='green', alpha=0.2, label='🟢 Alım Bölgesi')

        ax2.set_title('📊 RSI Momentum Göstergesi', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. WaveTrend Osilatör (Daha belirgin sinyaller)
        ax3 = axes[2]
        ax3.plot(self.data.index, self.data['WT1'], 
                label='🌊 WT1', linewidth=2, color='blue')
        ax3.plot(self.data.index, self.data['WT2'], 
                label='🌊 WT2', linewidth=2, color='red', alpha=0.7)

        # Yatay çizgiler
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='📊 Sıfır Çizgisi')
        ax3.axhline(y=self.data['WT_Overbought'].iloc[0], color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=self.data['WT_Oversold'].iloc[0], color='green', linestyle='--', alpha=0.7)

        # WaveTrend sinyallerini bul (veri çerçevesinde zaten hesaplanmış)
        buy_signals = self.data[self.data['WT_Buy_Signal']].index
        sell_signals = self.data[self.data['WT_Sell_Signal']].index
        zero_up = self.data[self.data['WT_Zero_Cross_Up']].index
        zero_down = self.data[self.data['WT_Zero_Cross_Down']].index

        # Sinyalleri çiz (daha büyük ve belirgin)
        if not buy_signals.empty:
            ax3.scatter(buy_signals, [self.data.loc[date, 'WT1'] for date in buy_signals],
            color='lime', marker='o', s=50, label='🟢 WT Al Sinyali', zorder=5)
    
        if not sell_signals.empty:
            ax3.scatter(sell_signals, [self.data.loc[date, 'WT1'] for date in sell_signals],
            color='red', marker='o', s=50, label='🔴 WT Sat Sinyali', zorder=5)
    
        if not zero_up.empty:
            ax3.scatter(zero_up, [0]*len(zero_up),
                   color='green', marker='^', s=100, label='🟢 Sıfır Yukarı', zorder=5)
    
        if not zero_down.empty:
            ax3.scatter(zero_down, [0]*len(zero_down),
                   color='red', marker='v', s=100, label='🔴 Sıfır Aşağı', zorder=5)

        # Bölge dolguları
        ax3.fill_between(self.data.index, self.data['WT1'], self.data['WT2'], 
                        where=(self.data['WT1'] > self.data['WT2']),
                        color='green', alpha=0.2, label='🟢 Boğa Bölgesi')
        ax3.fill_between(self.data.index, self.data['WT1'], self.data['WT2'],
                        where=(self.data['WT1'] <= self.data['WT2']),
                        color='red', alpha=0.2, label='🔴 Ayı Bölgesi')

        ax3.set_title('🌊 WaveTrend Osilatör - Net Sinyaller', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 4. Hacim analizi (Aynı kalacak)
        ax4 = axes[3]
        colors = ['green' if close > open_ else 'red' 
                 for close, open_ in zip(self.data['Close'], self.data['Open'])]
        ax4.bar(self.data.index, self.data['Volume'], 
               color=colors, alpha=0.6, width=0.8)
        ax4.set_title('📊 İşlem Hacmi', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Eksen formatı
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Grafik kaydedildi: {save_path}")
        plt.show(block=True)
    
    def run_strategy3_analysis(self):
        """Run complete Strategy 3 analysis with Turkish output"""
        print("🚀 Strateji 3 Analizi Başlatılıyor...")
        print("="*60)
        
        # Step 1: Fetch data
        success = self.fetch_data()
        if not success:
            print("⚠️ Canlı veri alınamadı, örnek veri kullanılıyor...")
            self.create_sample_data()
        
        # Step 2: Calculate indicators
        print("🔧 Teknik göstergeler hesaplanıyor...")
        self.calculate_indicators()
        
        # Step 3: Train LSTM
        print("🧠 LSTM modeli eğitiliyor...")
        lstm_history = self.train_lstm()
        
        # Step 4: Run backtest
        print("⏮️ Geçmiş performans analizi...")
        performance = self.backtest()
        
        # Step 5: Get current signal
        print("📡 Güncel sinyal analizi...")
        current_signal = self.get_current_signal_strategy3()
        
        # Step 6: Get Turkish recommendation
        recommendation = self.get_turkish_recommendation_strategy3()
        
        # Display results
        print("\n" + "="*60)
        print("📈 STRATEJİ ANALİZ SONUÇLARI")
        print("="*60)
        
        if 'error' not in performance:
            print(f"📊 Toplam İşlem: {performance['total_trades']}")
            print(f"💰 Toplam Kar: {performance['total_profit']:.2f}")
            print(f"📈 Ortalama Kar: {performance['avg_profit']:.2f}")
            print(f"🎯 Başarı Oranı: %{performance['win_rate']:.1f}")
            print(f"🚀 En Yüksek Kar: {performance['max_profit']:.2f}")
            print(f"📉 En Yüksek Zarar: {performance['max_loss']:.2f}")
        
        print(f"\n🔍 GÜNCEL DURUM:")
        print(f"⏰ Zaman: {current_signal['timestamp']}")
        print(f"💰 Güncel Fiyat: {recommendation['current_price']}")
        print(f"📊 RSI: {recommendation['rsi']}")
        print(f"📈 EMA: {recommendation['ema']}")
        print(f"🌊 WT1/WT2: {recommendation['wt1']}/{recommendation['wt2']}")
        print(f"🤖 LSTM Tahmin (Yüzde): {recommendation['lstm_prediction']}") 
        # LSTM tahmini fiyatı formatını koruyarak göster
        if current_signal['predicted_price'] is not None:
            current_price_str = recommendation['current_price']
            currency_symbol = ''.join([c for c in current_price_str if not c.isdigit() and c not in [',', '.']])
            formatted_prediction = f"{current_signal['predicted_price']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            print(f"🤖 LSTM Fiyat Tahmini: {currency_symbol}{formatted_prediction}")
            
        
        
        print(f"\n🎯 TAVSİYE:")
        print(f"🔥 {recommendation['action']}")
        print(f"📋 Durum: {recommendation['status']}")
        print(f"💡 Açıklama: {recommendation['explanation']}")
        print(f"🎯 Güven: {recommendation['confidence']}")
        print(f"⚠️ Risk Seviyesi: {recommendation['risk_level']}")
        
        # Plot analysis
        print(f"\n📊 Grafik analizi oluşturuluyor...")
        self.plot_analysis_strategy3()
        
        return {
            'performance': performance,
            'current_signal': current_signal,
            'recommendation': recommendation,
            'lstm_trained': lstm_history is not None
        }

def get_trading_symbol():
    """Get trading symbol from user with validation"""
    while True:
        symbol = input("🔍 Trading sembolünü girin (örn: BTCUSDT, ETHUSDT, XRPBTC): ").strip().upper()
        
        if not symbol:
            print("⚠️ Sembol boş olamaz! Lütfen geçerli bir sembol girin.")
            continue
        
        # Basic validation - check if it contains letters
        if not symbol.isalnum():
            print("⚠️ Sembol sadece harf ve rakam içermelidir!")
            continue
        
        # Check minimum length
        if len(symbol) < 6:
            print("⚠️ Sembol en az 6 karakter olmalıdır!")
            continue
        
        print(f"✅ Seçilen sembol: {symbol}")
        return symbol

def main():
    
    """Main function to run Strategy 3 analysis"""
    print("🎯 Kripto Trading Bot - Strateji 3")
    print("="*50)
    
    # Get trading symbol from user
    SYMBOL = get_trading_symbol()
    
    try:
        # Initialize bot with user-provided symbol
        bot = Strategy3TradingBot(symbol=SYMBOL, period="1h", lookback_days=100)
        
        # Run complete analysis
        results = bot.run_strategy3_analysis()
        
        print("\n✅ Analiz tamamlandı!")
        print("📊 Grafikleri kontrol edin ve tavsiyeyi değerlendirin.")
        
        return results
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()