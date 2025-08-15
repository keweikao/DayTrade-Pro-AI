"""
順勢交易策略
Trend Following Strategy - 趨勢是你的朋友
"""
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass

from ..models.stock_data import ComprehensiveStockData, MarketContext


class TrendFollowingStrategy:
    """順勢交易策略 - 基於專業交易員手冊"""
    
    def __init__(self):
        self.name = "順勢交易"
        self.philosophy = "趨勢是你的朋友，強者恆強，弱者恆弱"
        self.ideal_conditions = [
            "明確的單邊趨勢",
            "持續的動能確認", 
            "成交量配合",
            "技術指標同向"
        ]
        
        # 策略參數
        self.trend_confirmation_threshold = 0.02  # 趨勢確認閾值 2%
        self.volume_confirmation_ratio = 1.3      # 成交量確認倍數
        self.momentum_rsi_threshold = 50          # 動能RSI閾值
        self.risk_reward_target = 2.5             # 目標風險回報比
    
    def evaluate_suitability(self, stock_data: ComprehensiveStockData, 
                            market_context: MarketContext = None) -> Dict[str, Any]:
        """
        評估順勢策略適用性
        
        Args:
            stock_data: 完整股票數據
            market_context: 市場環境（可選）
            
        Returns:
            Dict: 策略適用性評估結果
        """
        
        score = 0
        conditions_met = []
        advantages = []
        disadvantages = []
        
        # 1. 趨勢強度分析 (權重: 35%)
        trend_analysis = self._analyze_trend_strength(stock_data)
        trend_score = trend_analysis['score']
        score += trend_score * 0.35
        
        if trend_analysis['direction'] != 'sideways':
            conditions_met.append(f"明確{trend_analysis['direction']}趨勢")
            advantages.append(f"趨勢方向明確({trend_analysis['direction']})")
            
            if trend_analysis['strength'] == 'strong':
                conditions_met.append("趨勢強勁")
                advantages.append("強勢趨勢，動能充足")
            elif trend_analysis['strength'] == 'moderate':
                conditions_met.append("趨勢中等")
        else:
            disadvantages.append("缺乏明確趨勢方向")
        
        # 2. 動能確認分析 (權重: 25%)
        momentum_analysis = self._analyze_momentum_signals(stock_data)
        momentum_score = momentum_analysis['score']
        score += momentum_score * 0.25
        
        for signal in momentum_analysis['signals']:
            conditions_met.append(signal)
            advantages.append(f"動能確認: {signal}")
        
        if len(momentum_analysis['signals']) < 2:
            disadvantages.append("動能信號不足")
        
        # 3. 成交量配合分析 (權重: 20%)
        volume_analysis = self._analyze_volume_confirmation(stock_data)
        volume_score = volume_analysis['score']
        score += volume_score * 0.20
        
        if volume_analysis['is_confirming']:
            conditions_met.append("成交量配合")
            advantages.append("成交量支撐趨勢")
        else:
            disadvantages.append("成交量不配合")
        
        # 4. 技術結構分析 (權重: 15%)
        structure_analysis = self._analyze_technical_structure(stock_data)
        structure_score = structure_analysis['score']
        score += structure_score * 0.15
        
        for feature in structure_analysis['positive_features']:
            conditions_met.append(feature)
            advantages.append(f"技術面: {feature}")
        
        for weakness in structure_analysis['weaknesses']:
            disadvantages.append(f"技術面: {weakness}")
        
        # 5. 市場環境適配 (權重: 5%)
        if market_context:
            market_score = self._analyze_market_environment(stock_data, market_context)
            score += market_score * 0.05
            
            if market_context.is_market_bullish and stock_data.price_vs_ma20 > 0:
                conditions_met.append("大盤環境配合")
                advantages.append("大盤趨勢一致")
            elif not market_context.is_market_bullish and stock_data.price_vs_ma20 < 0:
                conditions_met.append("大盤環境配合")
                advantages.append("空頭環境下的弱勢股")
            else:
                disadvantages.append("與大盤趨勢不一致")
        
        # 風險等級評估
        risk_level = self._assess_risk_level(trend_analysis, momentum_analysis)
        
        # 勝率與回報預期
        win_rate_expectation = self._estimate_win_rate(score, trend_analysis)
        risk_reward_expectation = self._estimate_risk_reward(trend_analysis, volume_analysis)
        
        return {
            'score': min(100, max(0, score)),
            'conditions_met': conditions_met,
            'advantages': advantages,
            'disadvantages': disadvantages,
            'risk_level': risk_level,
            'expected_win_rate': win_rate_expectation,
            'expected_risk_reward': risk_reward_expectation,
            'trend_analysis': trend_analysis,
            'momentum_analysis': momentum_analysis,
            'volume_analysis': volume_analysis
        }
    
    def _analyze_trend_strength(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析趨勢強度"""
        
        current_price = stock_data.current_price
        ma5 = stock_data.technical_indicators.ma5
        ma20 = stock_data.technical_indicators.ma20
        ma60 = stock_data.technical_indicators.ma60
        
        # 確定趨勢方向
        if ma5 > ma20 > ma60 and current_price > ma5:
            direction = 'uptrend'
            base_score = 80
        elif ma5 < ma20 < ma60 and current_price < ma5:
            direction = 'downtrend'
            base_score = 80
        elif ma5 > ma20 and current_price > ma20:
            direction = 'weak_uptrend'
            base_score = 60
        elif ma5 < ma20 and current_price < ma20:
            direction = 'weak_downtrend'
            base_score = 60
        else:
            direction = 'sideways'
            base_score = 30
        
        # 評估趨勢強度
        if direction in ['uptrend', 'downtrend']:
            # 計算均線傾斜度
            ma_slope = self._calculate_ma_slope(stock_data.price_history)
            
            if abs(ma_slope) > 0.02:  # 斜率大於2%
                strength = 'strong'
                strength_bonus = 20
            elif abs(ma_slope) > 0.01:  # 斜率大於1%
                strength = 'moderate'
                strength_bonus = 10
            else:
                strength = 'weak'
                strength_bonus = 0
        else:
            strength = 'none'
            strength_bonus = 0
        
        # 價格相對位置加分
        price_position_bonus = 0
        if direction == 'uptrend':
            price_vs_ma20 = stock_data.price_vs_ma20
            if price_vs_ma20 > 0.05:  # 高於MA20 5%以上
                price_position_bonus = 10
            elif price_vs_ma20 > 0.02:  # 高於MA20 2%以上
                price_position_bonus = 5
        elif direction == 'downtrend':
            price_vs_ma20 = stock_data.price_vs_ma20
            if price_vs_ma20 < -0.05:  # 低於MA20 5%以上
                price_position_bonus = 10
            elif price_vs_ma20 < -0.02:  # 低於MA20 2%以上
                price_position_bonus = 5
        
        final_score = base_score + strength_bonus + price_position_bonus
        
        return {
            'direction': direction,
            'strength': strength,
            'score': min(100, final_score),
            'ma_slope': ma_slope if 'ma_slope' in locals() else 0,
            'price_position': stock_data.price_vs_ma20
        }
    
    def _analyze_momentum_signals(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析動能信號"""
        
        indicators = stock_data.technical_indicators
        signals = []
        score = 0
        
        # RSI動能分析
        if indicators.rsi > self.momentum_rsi_threshold:
            if indicators.rsi > 60:
                signals.append("RSI強勢")
                score += 25
            else:
                signals.append("RSI偏強")
                score += 15
        elif indicators.rsi < (100 - self.momentum_rsi_threshold):
            if indicators.rsi < 40:
                signals.append("RSI弱勢")
                score += 25
            else:
                signals.append("RSI偏弱")
                score += 15
        
        # MACD動能分析
        if indicators.macd_bullish_crossover:
            signals.append("MACD黃金交叉")
            score += 30
        elif indicators.macd > 0 and indicators.macd_histogram > 0:
            signals.append("MACD多頭格局")
            score += 20
        elif indicators.macd < 0 and indicators.macd_histogram < 0:
            signals.append("MACD空頭格局")
            score += 20
        
        # KD動能分析
        if indicators.k > indicators.d:
            if indicators.k > 50:
                signals.append("KD多頭排列")
                score += 15
            else:
                signals.append("KD反彈")
                score += 10
        else:
            if indicators.k < 50:
                signals.append("KD空頭排列")
                score += 15
            else:
                signals.append("KD回檔")
                score += 10
        
        # 價格動能分析
        momentum_signals = stock_data.get_momentum_signals()
        for signal in momentum_signals:
            if signal not in [s.split(':')[0] if ':' in s else s for s in signals]:
                signals.append(signal)
                score += 10
        
        return {
            'score': min(100, score),
            'signals': signals,
            'momentum_strength': 'strong' if score > 70 else 'moderate' if score > 50 else 'weak'
        }
    
    def _analyze_volume_confirmation(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析成交量確認"""
        
        volume_ratio = stock_data.volume_ratio
        price_change = stock_data.price_vs_ma20
        
        # 量價配合分析
        is_confirming = False
        score = 0
        volume_type = ""
        
        if volume_ratio >= 2.0:
            volume_type = "暴量"
            score = 90
            is_confirming = True
        elif volume_ratio >= self.volume_confirmation_ratio:
            volume_type = "放量"
            score = 70
            is_confirming = True
        elif volume_ratio >= 1.0:
            volume_type = "正常量"
            score = 50
        else:
            volume_type = "縮量"
            score = 30
        
        # 量價背離檢查
        volume_price_divergence = False
        if volume_ratio > 1.5 and abs(price_change) < 0.01:  # 大量但價格平淡
            volume_price_divergence = True
            score -= 20
        elif volume_ratio < 0.8 and abs(price_change) > 0.03:  # 縮量但價格大幅變動
            volume_price_divergence = True
            score -= 15
        
        # 成交量趨勢分析
        volume_trend = self._analyze_volume_trend(stock_data.price_history)
        if volume_trend == 'increasing':
            score += 10
        elif volume_trend == 'decreasing':
            score -= 10
        
        return {
            'score': min(100, max(0, score)),
            'is_confirming': is_confirming,
            'volume_type': volume_type,
            'volume_ratio': volume_ratio,
            'volume_price_divergence': volume_price_divergence,
            'volume_trend': volume_trend
        }
    
    def _analyze_technical_structure(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析技術結構"""
        
        positive_features = []
        weaknesses = []
        score = 50  # 基礎分數
        
        # 支撐壓力分析
        support_resistance = stock_data.calculate_support_resistance()
        current_price = stock_data.current_price
        
        resistance = support_resistance['resistance']
        support = support_resistance['support']
        
        # 突破分析
        if current_price > resistance:
            positive_features.append("突破壓力位")
            score += 20
        elif current_price > resistance * 0.98:  # 接近壓力位
            positive_features.append("測試壓力位")
            score += 10
        
        if current_price < support:
            positive_features.append("跌破支撐位")  # 對空頭來說是正面的
            score += 15
        elif current_price < support * 1.02:  # 接近支撐位
            positive_features.append("測試支撐位")
            score += 10
        
        # 均線排列分析
        ma_trend = stock_data.ma_trend
        if ma_trend == 'bullish':
            positive_features.append("均線多頭排列")
            score += 15
        elif ma_trend == 'bearish':
            positive_features.append("均線空頭排列")
            score += 15
        else:
            weaknesses.append("均線排列混亂")
            score -= 10
        
        # 價格結構分析
        if len(stock_data.price_history) >= 5:
            recent_highs = [day.high for day in stock_data.price_history[-5:]]
            recent_lows = [day.low for day in stock_data.price_history[-5:]]
            
            # 高點抬升檢查
            if self._is_ascending_series(recent_highs[-3:]):
                positive_features.append("高點逐步抬升")
                score += 10
            
            # 低點抬升檢查
            if self._is_ascending_series(recent_lows[-3:]):
                positive_features.append("低點逐步抬升")
                score += 10
            
            # 高點下降檢查
            if self._is_descending_series(recent_highs[-3:]):
                positive_features.append("高點逐步下降")
                score += 10
            
            # 低點下降檢查
            if self._is_descending_series(recent_lows[-3:]):
                positive_features.append("低點逐步下降")
                score += 10
        
        # 波動性結構
        atr_pct = stock_data.atr_percentage
        if atr_pct > 0.03:
            positive_features.append("波動性充足")
            score += 5
        elif atr_pct < 0.015:
            weaknesses.append("波動性不足")
            score -= 10
        
        return {
            'score': min(100, max(0, score)),
            'positive_features': positive_features,
            'weaknesses': weaknesses
        }
    
    def _analyze_market_environment(self, stock_data: ComprehensiveStockData, 
                                   market_context: MarketContext) -> float:
        """分析市場環境適配性"""
        
        score = 50  # 基礎分數
        
        # 大盤趨勢一致性
        individual_trend = "up" if stock_data.price_vs_ma20 > 0 else "down"
        market_trend = "up" if market_context.is_market_bullish else "down"
        
        if individual_trend == market_trend:
            score += 30  # 趨勢一致
        else:
            score -= 20  # 趨勢不一致
        
        # 市場波動性環境
        if market_context.volatility_environment == "medium_volatility":
            score += 10  # 中等波動最適合順勢
        elif market_context.volatility_environment == "high_volatility":
            score -= 5   # 高波動增加風險
        elif market_context.volatility_environment == "low_volatility":
            score -= 10  # 低波動限制利潤空間
        
        # 交易時段適配
        if market_context.trading_session == "opening":
            score += 5   # 開盤適合追勢
        elif market_context.trading_session == "closing":
            score -= 15  # 收盤前風險高
        
        return min(100, max(0, score))
    
    def generate_execution_plan(self, stock_data: ComprehensiveStockData, 
                               market_context: MarketContext = None) -> Dict[str, Any]:
        """生成順勢交易執行計劃"""
        
        current_price = stock_data.current_price
        atr = stock_data.technical_indicators.atr
        ma20 = stock_data.technical_indicators.ma20
        
        # 確定趨勢方向
        trend_analysis = self._analyze_trend_strength(stock_data)
        trend_direction = trend_analysis['direction']
        
        if trend_direction in ['uptrend', 'weak_uptrend']:
            return self._generate_long_plan(stock_data, atr, ma20)
        elif trend_direction in ['downtrend', 'weak_downtrend']:
            return self._generate_short_plan(stock_data, atr, ma20)
        else:
            return self._generate_neutral_plan(stock_data)
    
    def _generate_long_plan(self, stock_data: ComprehensiveStockData, 
                           atr: float, ma20: float) -> Dict[str, Any]:
        """生成做多執行計劃"""
        
        current_price = stock_data.current_price
        support_resistance = stock_data.calculate_support_resistance()
        
        return {
            'direction': 'long',
            'entry_scenarios': [
                {
                    'scenario': '回檔進場',
                    'entry_price': max(ma20, current_price * 0.995),
                    'trigger': '價格回測20日均線附近獲得支撐',
                    'confidence': 75,
                    'wait_for_signals': ['成交量放大', '技術指標確認']
                },
                {
                    'scenario': '突破追價',
                    'entry_price': current_price + (atr * 0.3),
                    'trigger': '帶量突破前高或阻力位',
                    'confidence': 80,
                    'wait_for_signals': ['成交量暴增', '突破確認']
                },
                {
                    'scenario': '強勢拉升',
                    'entry_price': current_price * 1.005,
                    'trigger': '強勢拉升過程中的小幅回檔',
                    'confidence': 70,
                    'wait_for_signals': ['RSI未過熱', '量能持續']
                }
            ],
            'stop_loss': current_price - (atr * 1.5),
            'profit_targets': [
                {
                    'level': current_price + (atr * 2),
                    'percentage': 50,
                    'description': '第一目標：2倍ATR'
                },
                {
                    'level': current_price + (atr * 3),
                    'percentage': 30,
                    'description': '第二目標：3倍ATR'
                },
                {
                    'level': '讓利潤奔跑',
                    'percentage': 20,
                    'description': '持續持有直到趨勢反轉'
                }
            ],
            'risk_reward_ratio': f'1:{self.risk_reward_target}',
            'execution_tips': [
                "等待明確的進場信號，避免盲目追高",
                "進場必須有成交量配合",
                "大盤走勢要同向或至少不逆向",
                "設定移動停利保護利潤",
                "注意RSI是否過熱"
            ],
            'risk_management': {
                'max_risk_per_trade': '2%',
                'position_sizing': '根據停損距離計算',
                'trailing_stop': f'{atr * 1.2:.2f}元'
            }
        }
    
    def _generate_short_plan(self, stock_data: ComprehensiveStockData, 
                            atr: float, ma20: float) -> Dict[str, Any]:
        """生成做空執行計劃"""
        
        current_price = stock_data.current_price
        
        return {
            'direction': 'short',
            'entry_scenarios': [
                {
                    'scenario': '反彈空',
                    'entry_price': min(ma20, current_price * 1.005),
                    'trigger': '價格反彈至20日均線附近遇阻',
                    'confidence': 75,
                    'wait_for_signals': ['反彈無量', '技術指標轉弱']
                },
                {
                    'scenario': '跌破空',
                    'entry_price': current_price - (atr * 0.3),
                    'trigger': '帶量跌破前低或支撐位',
                    'confidence': 80,
                    'wait_for_signals': ['成交量放大', '跌破確認']
                }
            ],
            'stop_loss': current_price + (atr * 1.5),
            'profit_targets': [
                {
                    'level': current_price - (atr * 2),
                    'percentage': 50,
                    'description': '第一目標：2倍ATR'
                },
                {
                    'level': current_price - (atr * 3),
                    'percentage': 30,
                    'description': '第二目標：3倍ATR'
                },
                {
                    'level': '讓利潤奔跑',
                    'percentage': 20,
                    'description': '持續持有直到趨勢反轉'
                }
            ],
            'risk_reward_ratio': f'1:{self.risk_reward_target}',
            'execution_tips': [
                "空頭操作風險較高，更需嚴格紀律",
                "確認大盤也是弱勢",
                "注意不要在超賣區域盲目放空",
                "設定移動停利保護利潤"
            ]
        }
    
    def _generate_neutral_plan(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """生成中性市場執行計劃"""
        
        return {
            'direction': 'neutral',
            'recommendation': '觀望',
            'reason': '趨勢不明確，不適合順勢策略',
            'wait_for_conditions': [
                '明確的趨勢方向建立',
                '技術指標達成共識',
                '成交量配合趨勢發展'
            ],
            'alternative_strategies': ['counter_trend', 'breakout']
        }
    
    def _calculate_ma_slope(self, price_history: List) -> float:
        """計算均線斜率"""
        if len(price_history) < 10:
            return 0
        
        # 計算最近10天MA20的斜率
        recent_closes = [day.close for day in price_history[-10:]]
        
        # 簡單線性回歸計算斜率
        x = np.arange(len(recent_closes))
        y = np.array(recent_closes)
        
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        
        # 轉換為相對斜率
        return slope / np.mean(y) if np.mean(y) != 0 else 0
    
    def _analyze_volume_trend(self, price_history: List) -> str:
        """分析成交量趨勢"""
        if len(price_history) < 5:
            return 'unknown'
        
        recent_volumes = [day.volume for day in price_history[-5:]]
        
        # 計算趨勢
        if len(recent_volumes) >= 3:
            if recent_volumes[-1] > recent_volumes[-2] > recent_volumes[-3]:
                return 'increasing'
            elif recent_volumes[-1] < recent_volumes[-2] < recent_volumes[-3]:
                return 'decreasing'
        
        return 'stable'
    
    def _is_ascending_series(self, values: List[float]) -> bool:
        """檢查數列是否遞增"""
        return all(values[i] <= values[i+1] for i in range(len(values)-1))
    
    def _is_descending_series(self, values: List[float]) -> bool:
        """檢查數列是否遞減"""
        return all(values[i] >= values[i+1] for i in range(len(values)-1))
    
    def _assess_risk_level(self, trend_analysis: Dict, momentum_analysis: Dict) -> str:
        """評估風險等級"""
        
        if trend_analysis['strength'] == 'strong' and momentum_analysis['momentum_strength'] == 'strong':
            return '中低'
        elif trend_analysis['strength'] in ['strong', 'moderate'] and momentum_analysis['momentum_strength'] in ['strong', 'moderate']:
            return '中等'
        else:
            return '中高'
    
    def _estimate_win_rate(self, score: float, trend_analysis: Dict) -> str:
        """估計勝率"""
        
        if score >= 80 and trend_analysis['strength'] == 'strong':
            return '中高'
        elif score >= 70:
            return '中等'
        elif score >= 60:
            return '中等偏低'
        else:
            return '偏低'
    
    def _estimate_risk_reward(self, trend_analysis: Dict, volume_analysis: Dict) -> str:
        """估計風險回報比"""
        
        if (trend_analysis['strength'] == 'strong' and 
            volume_analysis['is_confirming']):
            return '高'
        elif trend_analysis['strength'] in ['strong', 'moderate']:
            return '中高'
        else:
            return '中等'