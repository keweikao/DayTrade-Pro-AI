"""
台股市場數據接口
Taiwan Market Data Interface
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
from dataclasses import dataclass
import random
import pytz

from ..models.stock_data import (
    StockPrice, TechnicalIndicators, MarketData, StockInfo, 
    ComprehensiveStockData, MarketContext
)


class TaiwanMarketDataProvider:
    """台股數據提供者 - 支援盤中與非盤中時段"""
    
    def __init__(self):
        self.tw_suffix = ".TW"  # 台股Yahoo Finance後綴
        self.otc_suffix = ".TWO"  # 櫃買中心後綴
        self.taiwan_tz = pytz.timezone('Asia/Taipei')
        
        # 台股產業分類映射
        self.industry_mapping = {
            '01': '水泥工業', '02': '食品工業', '03': '塑膠工業',
            '04': '紡織纖維', '05': '電機機械', '06': '電器電纜',
            '07': '化學生技醫療', '08': '玻璃陶瓷', '09': '造紙工業',
            '10': '鋼鐵工業', '11': '橡膠工業', '12': '汽車工業',
            '13': '半導體業', '14': '電腦及週邊設備業', '15': '光電業',
            '16': '通信網路業', '17': '電子零組件業', '18': '電子通路業',
            '19': '資訊服務業', '20': '其他電子業', '21': '建材營造業',
            '22': '航運業', '23': '觀光事業', '24': '金融保險業',
            '25': '貿易百貨業', '26': '綜合企業', '27': '其他業',
            '28': '油電燃氣業'
        }
        
        # 模擬數據生成器
        self.demo_mode = False
    
    def is_taiwan_market_open(self) -> bool:
        """檢查台股是否開盤"""
        try:
            now = datetime.now(self.taiwan_tz)
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            
            # 週末不開盤
            if weekday >= 5:
                return False
            
            # 開盤時間：平日 9:00-13:30
            current_time = now.time()
            market_open = datetime.strptime("09:00", "%H:%M").time()
            market_close = datetime.strptime("13:30", "%H:%M").time()
            
            return market_open <= current_time <= market_close
        except:
            # 如果時區處理失敗，假設非開盤時間
            return False
    
    def get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """獲取股票基本資訊 - 支援非開盤時間"""
        try:
            # 格式化股票代碼
            formatted_symbol = self._format_symbol(symbol)
            
            # 嘗試從Yahoo Finance獲取基本資訊
            try:
                ticker = yf.Ticker(formatted_symbol)
                info = ticker.info
                
                # 檢查是否獲取到有效數據
                if info and info.get('longName'):
                    industry = self._get_industry_from_symbol(symbol)
                    
                    stock_info = StockInfo(
                        code=symbol,
                        name=info.get('longName', f'股票{symbol}'),
                        industry=industry,
                        market_cap=info.get('marketCap', 0),
                        shares_outstanding=info.get('sharesOutstanding', 0),
                        price=info.get('currentPrice', 0)
                    )
                    
                    return stock_info
            except:
                pass
            
            # 如果Yahoo Finance失敗，使用模擬數據
            return self._generate_mock_stock_info(symbol)
            
        except Exception as e:
            print(f"獲取股票{symbol}基本資訊失敗: {e}")
            return self._generate_mock_stock_info(symbol)
    
    def _generate_mock_stock_info(self, symbol: str) -> StockInfo:
        """生成模擬股票基本資訊 (用於非開盤時間或API失敗)"""
        
        # 知名股票的模擬資料
        mock_data = {
            "2330": {"name": "台灣積體電路製造", "industry": "半導體業", "price": 590, "market_cap": 15000000000000, "shares": 25900000000},
            "2317": {"name": "鴻海精密", "industry": "電腦及週邊設備業", "price": 108, "market_cap": 1500000000000, "shares": 13900000000},
            "2454": {"name": "聯發科技", "industry": "半導體業", "price": 820, "market_cap": 1300000000000, "shares": 1580000000},
            "2308": {"name": "台達電子", "industry": "電子零組件業", "price": 320, "market_cap": 850000000000, "shares": 2650000000},
            "3008": {"name": "大立光電", "industry": "光電業", "price": 2800, "market_cap": 1200000000000, "shares": 430000000},
            "2880": {"name": "華南金控", "industry": "金融保險業", "price": 23.5, "market_cap": 320000000000, "shares": 13600000000},
            "2882": {"name": "國泰金控", "industry": "金融保險業", "price": 45.2, "market_cap": 720000000000, "shares": 15900000000},
            "2002": {"name": "中國鋼鐵", "industry": "鋼鐵工業", "price": 28.5, "market_cap": 280000000000, "shares": 9800000000},
            "1303": {"name": "南亞塑膠", "industry": "塑膠工業", "price": 78.5, "market_cap": 620000000000, "shares": 7900000000},
            "1301": {"name": "台塑", "industry": "塑膠工業", "price": 95.2, "market_cap": 750000000000, "shares": 7880000000},
        }
        
        if symbol in mock_data:
            data = mock_data[symbol]
            return StockInfo(
                code=symbol,
                name=data["name"],
                industry=data["industry"],
                market_cap=data["market_cap"],
                shares_outstanding=data["shares"],
                price=data["price"] * (0.95 + random.random() * 0.1)  # ±5% 隨機變動
            )
        else:
            # 一般股票的模擬數據
            industry = self._get_industry_from_symbol(symbol)
            base_price = 50 + random.random() * 200  # 50-250 元隨機價格
            
            return StockInfo(
                code=symbol,
                name=f"股票{symbol}",
                industry=industry,
                market_cap=int(base_price * 100000000 * (0.8 + random.random() * 0.4)),  # 隨機市值
                shares_outstanding=int(1000000000 * (0.5 + random.random())),  # 5億-15億股
                price=base_price
            )
    
    def get_price_history(self, symbol: str, days: int = 60) -> List[StockPrice]:
        """獲取股價歷史數據 - 支援非開盤時間"""
        try:
            formatted_symbol = self._format_symbol(symbol)
            
            # 嘗試從Yahoo Finance獲取歷史數據
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days + 30)
                
                ticker = yf.Ticker(formatted_symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    price_history = []
                    for date, row in hist.iterrows():
                        if pd.notna(row['Close']):
                            price_data = StockPrice(
                                date=date.to_pydatetime(),
                                open=float(row['Open']),
                                high=float(row['High']),
                                low=float(row['Low']),
                                close=float(row['Close']),
                                volume=int(row['Volume'])
                            )
                            price_history.append(price_data)
                    
                    # 返回最近的指定天數
                    result = price_history[-days:] if len(price_history) > days else price_history
                    if result:
                        return result
            except:
                pass
            
            # 如果Yahoo Finance失敗，生成模擬歷史數據
            return self._generate_mock_price_history(symbol, days)
            
        except Exception as e:
            print(f"獲取股票{symbol}歷史數據失敗: {e} - 使用模擬數據")
            return self._generate_mock_price_history(symbol, days)
    
    def _generate_mock_price_history(self, symbol: str, days: int) -> List[StockPrice]:
        """生成模擬股價歷史數據"""
        
        # 獲取基礎價格
        stock_info = self._generate_mock_stock_info(symbol)
        base_price = stock_info.price
        
        price_history = []
        current_price = base_price
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            
            # 模擬價格變動 (±3% 隨機變動)
            daily_change = 1 + (random.random() - 0.5) * 0.06
            current_price *= daily_change
            
            # 確保價格合理範圍
            current_price = max(base_price * 0.7, min(base_price * 1.3, current_price))
            
            # 生成開高低收
            daily_volatility = 0.02 + random.random() * 0.03  # 2-5% 日內波動
            open_price = current_price * (1 + (random.random() - 0.5) * daily_volatility)
            
            high_price = max(open_price, current_price) * (1 + random.random() * daily_volatility)
            low_price = min(open_price, current_price) * (1 - random.random() * daily_volatility)
            
            # 模擬成交量
            base_volume = 1000000 + random.randint(0, 5000000)  # 100萬-600萬股
            
            price_data = StockPrice(
                date=date,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(current_price, 2),
                volume=base_volume
            )
            
            price_history.append(price_data)
        
        return price_history
    
    def get_current_market_data(self, symbol: str) -> Optional[MarketData]:
        """獲取當前市場數據 - 支援非開盤時間"""
        try:
            formatted_symbol = self._format_symbol(symbol)
            
            # 檢查是否開盤時間
            is_market_open = self.is_taiwan_market_open()
            
            # 嘗試獲取實時數據
            try:
                ticker = yf.Ticker(formatted_symbol)
                
                if is_market_open:
                    # 開盤時間嘗試獲取1分鐘數據
                    hist = ticker.history(period="1d", interval="1m")
                else:
                    # 非開盤時間獲取最近的日線數據
                    hist = ticker.history(period="2d", interval="1d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    last_price = float(latest['Close'])
                    estimated_spread = last_price * 0.001
                    
                    market_data = MarketData(
                        bid_price=last_price - estimated_spread/2,
                        ask_price=last_price + estimated_spread/2,
                        bid_size=100,
                        ask_size=100,
                        last_price=last_price,
                        volume=int(latest['Volume']),
                        timestamp=datetime.now()
                    )
                    
                    return market_data
            except:
                pass
            
            # 如果API失敗，使用模擬數據
            return self._generate_mock_market_data(symbol)
            
        except Exception as e:
            print(f"獲取股票{symbol}實時數據失敗: {e} - 使用模擬數據")
            return self._generate_mock_market_data(symbol)
    
    def _generate_mock_market_data(self, symbol: str) -> MarketData:
        """生成模擬市場數據"""
        
        # 獲取基礎價格
        stock_info = self._generate_mock_stock_info(symbol)
        base_price = stock_info.price
        
        # 添加小幅隨機變動
        current_price = base_price * (0.98 + random.random() * 0.04)  # ±2% 變動
        
        # 估計買賣價差
        spread_pct = 0.001 + random.random() * 0.002  # 0.1%-0.3% 價差
        spread = current_price * spread_pct
        
        # 模擬成交量
        if self.is_taiwan_market_open():
            base_volume = 10000 + random.randint(0, 100000)  # 開盤時較大量
        else:
            base_volume = 1000 + random.randint(0, 10000)   # 非開盤時較小量
        
        return MarketData(
            bid_price=round(current_price - spread/2, 2),
            ask_price=round(current_price + spread/2, 2),
            bid_size=random.randint(10, 500) * 100,  # 1千-5萬股
            ask_size=random.randint(10, 500) * 100,
            last_price=round(current_price, 2),
            volume=base_volume,
            timestamp=datetime.now()
        )
    
    def calculate_technical_indicators(self, price_history: List[StockPrice]) -> Optional[TechnicalIndicators]:
        """計算技術指標"""
        if len(price_history) < 20:
            return None
        
        try:
            # 轉換為DataFrame方便計算
            df = pd.DataFrame([
                {
                    'Date': p.date,
                    'Open': p.open,
                    'High': p.high,
                    'Low': p.low,
                    'Close': p.close,
                    'Volume': p.volume
                }
                for p in price_history
            ])
            
            df.set_index('Date', inplace=True)
            
            # 計算各種技術指標
            indicators = TechnicalIndicators(
                rsi=self._calculate_rsi(df['Close'].values),
                macd=0,  # 將在下面計算
                macd_signal=0,
                macd_histogram=0,
                k=0,  # 將在下面計算
                d=0,
                ma5=df['Close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else df['Close'].iloc[-1],
                ma20=df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Close'].iloc[-1],
                ma60=df['Close'].rolling(60).mean().iloc[-1] if len(df) >= 60 else df['Close'].iloc[-1],
                atr=self._calculate_atr(df)
            )
            
            # 計算MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(df['Close'].values)
            indicators.macd = macd_line
            indicators.macd_signal = macd_signal
            indicators.macd_histogram = macd_histogram
            
            # 計算KD
            k, d = self._calculate_kd(df)
            indicators.k = k
            indicators.d = d
            
            return indicators
            
        except Exception as e:
            print(f"計算技術指標失敗: {e}")
            return None
    
    def get_comprehensive_stock_data(self, symbol: str) -> Optional[ComprehensiveStockData]:
        """獲取完整股票數據"""
        try:
            # 獲取各項數據
            stock_info = self.get_stock_info(symbol)
            if not stock_info:
                return None
            
            price_history = self.get_price_history(symbol)
            if not price_history:
                return None
            
            current_market = self.get_current_market_data(symbol)
            if not current_market:
                return None
            
            technical_indicators = self.calculate_technical_indicators(price_history)
            if not technical_indicators:
                return None
            
            # 更新股票資訊中的現價
            stock_info.price = current_market.last_price
            
            comprehensive_data = ComprehensiveStockData(
                info=stock_info,
                current_market=current_market,
                price_history=price_history,
                technical_indicators=technical_indicators
            )
            
            return comprehensive_data
            
        except Exception as e:
            print(f"獲取股票{symbol}完整數據失敗: {e}")
            return None
    
    def get_market_context(self) -> MarketContext:
        """獲取市場環境數據"""
        try:
            # 獲取台股指數數據
            taiex_ticker = yf.Ticker("^TWII")
            taiex_data = taiex_ticker.history(period="2d")
            
            if len(taiex_data) >= 2:
                current_taiex = float(taiex_data['Close'].iloc[-1])
                previous_taiex = float(taiex_data['Close'].iloc[-2])
                taiex_change = current_taiex - previous_taiex
                taiex_change_pct = taiex_change / previous_taiex
            else:
                current_taiex = 17000  # 預設值
                taiex_change = 0
                taiex_change_pct = 0
            
            # 模擬VIX（實際需要從相關數據源獲取）
            vix = 20 + np.random.normal(0, 3)  # 模擬VIX數據
            
            # 模擬外資買賣超
            foreign_investment = np.random.normal(0, 50)  # 模擬外資數據
            
            # 模擬族群表現
            sector_performance = {
                '半導體業': np.random.normal(0, 0.02),
                '電子零組件業': np.random.normal(0, 0.02),
                '光電業': np.random.normal(0, 0.02),
                '金融保險業': np.random.normal(0, 0.015),
                '塑膠工業': np.random.normal(0, 0.015)
            }
            
            # 判斷市場情緒
            if taiex_change_pct > 0.01:
                sentiment = "bullish"
            elif taiex_change_pct < -0.01:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            # 判斷交易時段
            current_hour = datetime.now().hour
            if 9 <= current_hour < 13:
                if current_hour == 9:
                    trading_session = "opening"
                elif current_hour >= 12:
                    trading_session = "closing"
                else:
                    trading_session = "trading"
            else:
                trading_session = "closed"
            
            market_context = MarketContext(
                taiex=current_taiex,
                taiex_change=taiex_change,
                taiex_change_pct=taiex_change_pct,
                vix=vix,
                foreign_investment=foreign_investment,
                sector_performance=sector_performance,
                market_sentiment=sentiment,
                trading_session=trading_session
            )
            
            return market_context
            
        except Exception as e:
            print(f"獲取市場環境數據失敗: {e}")
            # 返回預設市場環境
            return MarketContext(
                taiex=17000,
                taiex_change=0,
                taiex_change_pct=0,
                vix=20,
                foreign_investment=0,
                sector_performance={},
                market_sentiment="neutral",
                trading_session="closed"
            )
    
    def _format_symbol(self, symbol: str) -> str:
        """格式化股票代碼"""
        # 移除可能的後綴
        clean_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        
        # 根據代碼判斷市場
        if clean_symbol.startswith('00') or len(clean_symbol) == 4:
            # 主板或ETF
            return f"{clean_symbol}.TW"
        else:
            # 櫃買中心
            return f"{clean_symbol}.TWO"
    
    def _get_industry_from_symbol(self, symbol: str) -> str:
        """根據股票代碼推斷產業"""
        # 簡化的產業分類（實際應該查詢資料庫或API）
        if symbol.startswith('23'):
            return '半導體業'
        elif symbol.startswith('24'):
            return '電腦及週邊設備業'
        elif symbol.startswith('25'):
            return '光電業'
        elif symbol.startswith('26'):
            return '通信網路業'
        elif symbol.startswith('27'):
            return '電子零組件業'
        elif symbol.startswith('28'):
            return '金融保險業'
        else:
            return '其他業'
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """計算RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """計算MACD"""
        if len(prices) < slow:
            return 0, 0, 0
        
        # 計算EMA
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        # MACD線
        macd_line = ema_fast - ema_slow
        
        # 計算MACD的EMA作為信號線
        macd_history = []
        for i in range(slow-1, len(prices)):
            fast_ema = self._calculate_ema(prices[:i+1], fast)
            slow_ema = self._calculate_ema(prices[:i+1], slow)
            macd_history.append(fast_ema - slow_ema)
        
        if len(macd_history) >= signal:
            signal_line = self._calculate_ema(np.array(macd_history), signal)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        return float(macd_line), float(signal_line), float(histogram)
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """計算指數移動平均"""
        if len(prices) == 0:
            return 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)
    
    def _calculate_kd(self, df: pd.DataFrame, k_period: int = 9, d_period: int = 3) -> Tuple[float, float]:
        """計算KD指標"""
        if len(df) < k_period:
            return 50, 50
        
        # 計算最近k_period天的最高價和最低價
        high_max = df['High'].rolling(k_period).max()
        low_min = df['Low'].rolling(k_period).min()
        
        # 計算RSV
        rsv = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        rsv = rsv.fillna(50)
        
        # 計算K值（RSV的移動平均）
        k_values = []
        k = 50  # 初始值
        
        for rsv_val in rsv:
            if pd.notna(rsv_val):
                k = (2/3) * k + (1/3) * rsv_val
                k_values.append(k)
        
        # 計算D值（K值的移動平均）
        d_values = []
        d = 50  # 初始值
        
        for k_val in k_values:
            d = (2/3) * d + (1/3) * k_val
            d_values.append(d)
        
        return float(k_values[-1]) if k_values else 50, float(d_values[-1]) if d_values else 50
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """計算平均真實波幅"""
        if len(df) < 2:
            return 0
        
        # 計算真實波幅
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # 計算ATR
        if len(true_range) >= period:
            atr = true_range.rolling(period).mean().iloc[-1]
        else:
            atr = true_range.mean()
        
        return float(atr) if pd.notna(atr) else 0


class TaiwanStockScreener:
    """台股篩選器"""
    
    def __init__(self):
        self.data_provider = TaiwanMarketDataProvider()
        
        # 台股熱門股票池（可以擴展）
        self.stock_universe = [
            '2330', '2317', '2454', '2308', '3008',  # 科技股
            '2412', '2882', '2891', '2886', '2880',  # 金融股
            '1303', '1326', '1301', '6505', '2002',  # 傳產股
            '2379', '3711', '4938', '6770', '3034'   # 其他熱門股
        ]
    
    def screen_daytrading_candidates(self, criteria: Dict = None) -> List[str]:
        """篩選當沖候選股票"""
        if criteria is None:
            criteria = {
                'min_volume': 10000,      # 最小成交量
                'min_price': 20,          # 最低股價
                'max_price': 500,         # 最高股價
                'min_market_cap': 5000000000  # 最小市值
            }
        
        candidates = []
        
        for symbol in self.stock_universe:
            try:
                stock_info = self.data_provider.get_stock_info(symbol)
                if not stock_info:
                    continue
                
                # 應用篩選條件
                if (stock_info.price >= criteria['min_price'] and
                    stock_info.price <= criteria['max_price'] and
                    stock_info.market_cap >= criteria['min_market_cap']):
                    
                    candidates.append(symbol)
                
                # 避免請求過於頻繁
                time.sleep(0.1)
                
            except Exception as e:
                print(f"篩選股票{symbol}時發生錯誤: {e}")
                continue
        
        return candidates