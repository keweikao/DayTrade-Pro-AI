# Trading Strategy Modules

from .strategy_identifier import StrategyIdentifier
from .trend_following import TrendFollowingStrategy
from .counter_trend import CounterTrendStrategy
from .breakout_strategy import BreakoutStrategy

__all__ = [
    'StrategyIdentifier',
    'TrendFollowingStrategy', 
    'CounterTrendStrategy',
    'BreakoutStrategy'
]