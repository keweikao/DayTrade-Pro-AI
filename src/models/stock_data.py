"""
股票數據模型
Stock Data Models for Taiwan Market
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np


@dataclass
class StockPrice:
    """單日股價數據"""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    @property
    def range_percentage(self) -> float:
        """日振幅百分比"""
        if self.open == 0:
            return 0
        return (self.high - self.low) / self.open


@dataclass
class TechnicalIndicators:
    """技術指標數據"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    k: float  # KD指標K值
    d: float  # KD指標D值
    ma5: float
    ma20: float
    ma60: float
    atr: float
    
    @property
    def macd_bullish_crossover(self) -> bool:
        """MACD黃金交叉"""
        return self.macd > self.macd_signal and self.macd_histogram > 0
    
    @property
    def kd_oversold_golden_cross(self) -> bool:
        """KD超賣區黃金交叉"""
        return self.k < 30 and self.d < 30 and self.k > self.d


@dataclass
class MarketData:
    """市場數據"""
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    timestamp: datetime
    
    @property
    def bid_ask_spread(self) -> float:
        """買賣價差"""
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        """中間價"""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread_percentage(self) -> float:
        """價差百分比"""
        if self.mid_price == 0:
            return 0
        return self.bid_ask_spread / self.mid_price


@dataclass
class StockInfo:
    """股票基本資訊"""
    code: str
    name: str
    industry: str
    market_cap: float  # 市值(億元)
    shares_outstanding: int  # 流通股數
    price: float
    
    @property
    def market_cap_billions(self) -> float:
        """市值(億元)"""
        return self.market_cap / 100000000


@dataclass
class ComprehensiveStockData:
    """完整股票數據"""
    info: StockInfo
    current_market: MarketData
    price_history: List[StockPrice]
    technical_indicators: TechnicalIndicators
    
    # 計算屬性
    _avg_volume_cache: Optional[float] = None
    _turnover_rate_cache: Optional[float] = None
    _volatility_cache: Optional[float] = None
    
    @property
    def symbol(self) -> str:
        """股票代號（向後兼容性）"""
        return self.info.code
    
    @property
    def current_price(self) -> float:
        """現價"""
        return self.current_market.last_price
    
    @property
    def avg_volume_20d(self) -> float:
        """20日平均成交量"""
        if self._avg_volume_cache is None:
            recent_volumes = [day.volume for day in self.price_history[-20:]]
            self._avg_volume_cache = np.mean(recent_volumes) if recent_volumes else 0
        return self._avg_volume_cache
    
    @property
    def turnover_rate(self) -> float:
        """週轉率"""
        if self._turnover_rate_cache is None:
            if self.info.shares_outstanding > 0:
                daily_volume = self.current_market.volume
                self._turnover_rate_cache = (daily_volume * 1000) / self.info.shares_outstanding
            else:
                self._turnover_rate_cache = 0
        return self._turnover_rate_cache
    
    @property
    def avg_range_5d(self) -> float:
        """近5日平均振幅"""
        if len(self.price_history) < 5:
            return 0
        
        recent_ranges = [day.range_percentage for day in self.price_history[-5:]]
        return np.mean(recent_ranges)
    
    @property
    def atr_percentage(self) -> float:
        """ATR百分比"""
        if self.current_price == 0:
            return 0
        return self.technical_indicators.atr / self.current_price
    
    @property
    def volume_ratio(self) -> float:
        """成交量比率"""
        if self.avg_volume_20d == 0:
            return 0
        return self.current_market.volume / self.avg_volume_20d
    
    @property
    def price_vs_ma20(self) -> float:
        """相對20日均線位置"""
        if self.technical_indicators.ma20 == 0:
            return 0
        return (self.current_price / self.technical_indicators.ma20) - 1
    
    @property
    def ma_trend(self) -> str:
        """均線趨勢"""
        ma5 = self.technical_indicators.ma5
        ma20 = self.technical_indicators.ma20
        ma60 = self.technical_indicators.ma60
        
        if ma5 > ma20 > ma60:
            return "bullish"
        elif ma5 < ma20 < ma60:
            return "bearish"
        else:
            return "mixed"
    
    def calculate_support_resistance(self) -> Dict[str, float]:
        """計算支撐壓力位"""
        if len(self.price_history) < 20:
            return {"support": self.current_price * 0.95, "resistance": self.current_price * 1.05}
        
        recent_highs = [day.high for day in self.price_history[-20:]]
        recent_lows = [day.low for day in self.price_history[-20:]]
        
        # 簡化的支撐壓力計算
        resistance = np.percentile(recent_highs, 80)
        support = np.percentile(recent_lows, 20)
        
        return {
            "support": support,
            "resistance": resistance
        }
    
    def has_breakout_signal(self) -> bool:
        """是否有突破信號"""
        support_resistance = self.calculate_support_resistance()
        
        # 價格突破壓力位且有量配合
        price_breakout = self.current_price > support_resistance["resistance"]
        volume_confirmation = self.volume_ratio > 1.3
        
        return price_breakout and volume_confirmation
    
    def get_momentum_signals(self) -> List[str]:
        """獲取動能信號"""
        signals = []
        
        # 技術指標信號
        if self.technical_indicators.macd_bullish_crossover:
            signals.append("MACD黃金交叉")
            
        if self.technical_indicators.kd_oversold_golden_cross:
            signals.append("KD超賣反彈")
            
        if self.technical_indicators.rsi > 50 and self.technical_indicators.rsi < 80:
            signals.append("RSI強勢")
            
        # 均線信號
        if self.ma_trend == "bullish":
            signals.append("均線多頭排列")
            
        # 量價信號
        if self.volume_ratio > 1.5 and self.price_vs_ma20 > 0:
            signals.append("量價配合突破")
            
        # 突破信號
        if self.has_breakout_signal():
            signals.append("價格突破")
            
        return signals


@dataclass
class MarketContext:
    """市場環境數據"""
    taiex: float
    taiex_change: float
    taiex_change_pct: float
    vix: float
    foreign_investment: float  # 外資買賣超(億元)
    sector_performance: Dict[str, float]
    market_sentiment: str  # "bullish", "bearish", "neutral"
    trading_session: str  # "pre_market", "opening", "trading", "closing"
    
    @property
    def is_market_bullish(self) -> bool:
        """市場是否看多"""
        return self.taiex_change_pct > 0.005  # 上漲0.5%以上視為看多
    
    @property
    def volatility_environment(self) -> str:
        """波動性環境"""
        if self.vix > 25:
            return "high_volatility"
        elif self.vix > 20:
            return "medium_volatility"
        else:
            return "low_volatility"


@dataclass
class AnalysisResult:
    """分析結果基類"""
    score: float
    grade: str
    details: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class LiquidityAnalysis(AnalysisResult):
    """流動性分析結果"""
    is_tradeable: bool = True
    volume_score: float = 0.0
    spread_score: float = 0.0
    depth_score: float = 0.0


@dataclass
class VolatilityAnalysis(AnalysisResult):
    """波動性分析結果"""
    atr_percentage: float = 0.0
    avg_daily_range: float = 0.0
    opportunity_level: str = "medium"
    volatility_trend: str = "stable"


@dataclass
class MomentumAnalysis(AnalysisResult):
    """動能分析結果"""
    catalysts: List[str] = None
    momentum_grade: str = "C"
    sustainability: str = "medium"
    momentum_score: float = 50.0
    
    def __post_init__(self):
        super().__post_init__()
        if self.catalysts is None:
            self.catalysts = []


@dataclass
class StrategyRecommendation:
    """策略推薦"""
    strategy_name: str = ""
    strategy_score: float = 0.0
    entry_scenarios: List[Dict[str, Any]] = None
    risk_reward_ratio: str = "1:1"
    execution_tips: List[str] = None
    risk_level: str = "medium"
    
    def __post_init__(self):
        if self.entry_scenarios is None:
            self.entry_scenarios = []
        if self.execution_tips is None:
            self.execution_tips = []


@dataclass
class RiskAssessment:
    """風險評估"""
    max_position_size: int = 0  # 最大股數
    recommended_stop_loss: float = 0.0
    risk_amount: float = 0.0
    risk_percentage: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class FundAllocation:
    """資金配置"""
    recommended_lots: int = 0  # 建議張數
    investment_amount: float = 0.0
    risk_amount: float = 0.0
    stop_loss_price: float = 0.0
    profit_targets: List[Dict[str, float]] = None
    trading_cost: Dict[str, float] = None
    
    def __post_init__(self):
        if self.profit_targets is None:
            self.profit_targets = []
        if self.trading_cost is None:
            self.trading_cost = {}


@dataclass
class ComprehensiveRecommendation:
    """完整推薦"""
    stock_data: ComprehensiveStockData = None
    liquidity_analysis: LiquidityAnalysis = None
    volatility_analysis: VolatilityAnalysis = None
    momentum_analysis: MomentumAnalysis = None
    strategy_recommendation: StrategyRecommendation = None
    risk_assessment: RiskAssessment = None
    fund_allocation: FundAllocation = None
    overall_score: float = 0.0
    recommendation_level: str = "hold"  # "strong_buy", "buy", "hold", "avoid"