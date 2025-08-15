"""
突破交易策略
Breakout Strategy - 捕捉動能爆發的瞬間
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..models.stock_data import ComprehensiveStockData, MarketContext


class BreakoutStrategy:
    """突破交易策略 - 基於專業交易員手冊"""
    
    def __init__(self):
        self.name = "突破交易"
        self.philosophy = "捕捉動能爆發的瞬間，在市場共識改變時獲利"
        self.ideal_conditions = [
            "整理後的突破",
            "成交量確認",
            "技術型態完成",
            "關鍵位置突破"
        ]
        
        # 策略參數
        self.breakout_threshold = 0.015           # 突破閾值 1.5%
        self.volume_confirmation_ratio = 2.0      # 突破確認成交量倍數
        self.consolidation_min_days = 3           # 最小整理天數
        self.consolidation_max_days = 20          # 最大整理天數
        self.volatility_compression_threshold = 0.7  # 波動性壓縮閾值
        self.false_breakout_threshold = 0.005     # 假突破閾值 0.5%
    
    def evaluate_suitability(self, stock_data: ComprehensiveStockData, 
                            market_context: MarketContext = None) -> Dict[str, Any]:
        """
        評估突破策略適用性
        
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
        
        # 1. 整理型態分析 (權重: 35%)
        consolidation_analysis = self._analyze_consolidation_pattern(stock_data)
        consolidation_score = consolidation_analysis['score']
        score += consolidation_score * 0.35
        
        if consolidation_analysis['has_pattern']:
            conditions_met.append(f"整理型態: {consolidation_analysis['pattern_type']}")
            advantages.append(f"形成{consolidation_analysis['pattern_type']}，蓄勢待發")
            
            if consolidation_analysis['pattern_quality'] == 'high':
                conditions_met.append("高品質整理型態")
                advantages.append("整理型態品質優秀，突破機率高")
            elif consolidation_analysis['pattern_quality'] == 'medium':
                conditions_met.append("中等品質整理型態")
                advantages.append("整理型態尚可，有突破潛力")
        else:
            disadvantages.append("缺乏明確的整理型態")
        
        # 2. 突破確認分析 (權重: 25%)
        breakout_analysis = self._analyze_breakout_confirmation(stock_data)
        breakout_score = breakout_analysis['score']
        score += breakout_score * 0.25
        
        if breakout_analysis['confirmed']:
            conditions_met.append(f"突破確認: {breakout_analysis['direction']}")
            advantages.append(f"已確認{breakout_analysis['direction']}突破")
            
            if breakout_analysis['strength'] == 'strong':
                conditions_met.append("強力突破")
                advantages.append("突破力道強勁，後續空間大")
            elif breakout_analysis['strength'] == 'moderate':
                conditions_met.append("中等強度突破")
        else:
            disadvantages.append("尚未確認有效突破")
        
        # 3. 成交量爆發分析 (權重: 20%)
        volume_analysis = self._analyze_volume_explosion(stock_data)
        volume_score = volume_analysis['score']
        score += volume_score * 0.20
        
        if volume_analysis['confirmed']:
            volume_ratio = volume_analysis['ratio']
            conditions_met.append(f"成交量爆發: {volume_ratio:.1f}倍")
            advantages.append(f"成交量暴增{volume_ratio:.1f}倍，確認突破有效")
        else:
            disadvantages.append("成交量未能配合突破")
        
        # 4. 波動性分析 (權重: 15%)
        volatility_analysis = self._analyze_volatility_pattern(stock_data)
        volatility_score = volatility_analysis['score']
        score += volatility_score * 0.15
        
        if volatility_analysis['compression_detected']:
            conditions_met.append("波動性壓縮")
            advantages.append("波動性收斂，蓄積突破能量")
        
        if volatility_analysis['expansion_started']:
            conditions_met.append("波動性擴張")
            advantages.append("波動性開始擴張，突破行情啟動")
        
        # 5. 市場環境分析 (權重: 5%)
        if market_context:
            market_score = self._analyze_market_environment_for_breakout(stock_data, market_context)
            score += market_score * 0.05
            
            if market_context.volatility_environment in ["low_volatility", "medium_volatility"]:
                conditions_met.append("適合突破的波動環境")
                advantages.append("市場波動環境有利突破交易")
            else:
                disadvantages.append("高波動環境增加假突破風險")
        
        # 風險等級評估
        risk_level = self._assess_breakout_risk_level(consolidation_analysis, breakout_analysis)
        
        # 勝率與回報預期
        win_rate_expectation = self._estimate_breakout_win_rate(score, breakout_analysis)
        risk_reward_expectation = self._estimate_breakout_risk_reward(consolidation_analysis, breakout_analysis)
        
        # 檢查假突破風險
        false_breakout_risks = self._identify_false_breakout_risks(stock_data, market_context)
        disadvantages.extend(false_breakout_risks)
        
        return {
            'score': min(100, max(0, score)),
            'conditions_met': conditions_met,
            'advantages': advantages,
            'disadvantages': disadvantages,
            'risk_level': risk_level,
            'expected_win_rate': win_rate_expectation,
            'expected_risk_reward': risk_reward_expectation,
            'consolidation_analysis': consolidation_analysis,
            'breakout_analysis': breakout_analysis,
            'volume_analysis': volume_analysis,
            'false_breakout_risks': false_breakout_risks
        }
    
    def _analyze_consolidation_pattern(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析整理型態"""
        
        if len(stock_data.price_history) < self.consolidation_min_days:
            return {
                'has_pattern': False,
                'pattern_type': 'insufficient_data',
                'score': 20,
                'pattern_quality': 'poor'
            }
        
        recent_data = stock_data.price_history[-self.consolidation_max_days:]
        highs = [day.high for day in recent_data]
        lows = [day.low for day in recent_data]
        closes = [day.close for day in recent_data]
        
        # 檢測不同類型的整理型態
        pattern_results = {
            'rectangle': self._detect_rectangle_pattern(highs, lows),
            'triangle': self._detect_triangle_pattern(highs, lows),
            'flag': self._detect_flag_pattern(closes),
            'wedge': self._detect_wedge_pattern(highs, lows)
        }
        
        # 找出最佳型態
        best_pattern = max(pattern_results.items(), key=lambda x: x[1]['score'])
        pattern_name, pattern_data = best_pattern
        
        if pattern_data['score'] > 60:
            has_pattern = True
            pattern_type = pattern_name
            
            # 評估型態品質
            if pattern_data['score'] > 80:
                quality = 'high'
            elif pattern_data['score'] > 70:
                quality = 'medium'
            else:
                quality = 'low'
        else:
            has_pattern = False
            pattern_type = 'no_clear_pattern'
            quality = 'poor'
        
        # 計算整理時間
        consolidation_days = self._calculate_consolidation_days(stock_data)
        
        # 計算價格壓縮程度
        price_compression = self._calculate_price_compression(recent_data)
        
        return {
            'has_pattern': has_pattern,
            'pattern_type': pattern_type,
            'pattern_quality': quality,
            'score': pattern_data['score'],
            'consolidation_days': consolidation_days,
            'price_compression': price_compression,
            'pattern_details': pattern_data
        }
    
    def _analyze_breakout_confirmation(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析突破確認"""
        
        current_price = stock_data.current_price
        support_resistance = stock_data.calculate_support_resistance()
        
        resistance = support_resistance['resistance']
        support = support_resistance['support']
        
        confirmed = False
        direction = 'none'
        strength = 'none'
        score = 30  # 基礎分數
        
        # 檢查向上突破
        upward_breakout_pct = (current_price - resistance) / resistance
        if upward_breakout_pct > self.breakout_threshold:
            confirmed = True
            direction = '向上'
            
            if upward_breakout_pct > 0.03:  # 3%以上突破
                strength = 'strong'
                score = 90
            elif upward_breakout_pct > self.breakout_threshold:
                strength = 'moderate'
                score = 75
        
        # 檢查向下突破
        downward_breakout_pct = (support - current_price) / support
        if downward_breakout_pct > self.breakout_threshold:
            confirmed = True
            direction = '向下'
            
            if downward_breakout_pct > 0.03:  # 3%以上突破
                strength = 'strong'
                score = 90
            elif downward_breakout_pct > self.breakout_threshold:
                strength = 'moderate'
                score = 75
        
        # 檢查是否接近突破點
        if not confirmed:
            resistance_distance = abs(current_price - resistance) / resistance
            support_distance = abs(current_price - support) / support
            
            min_distance = min(resistance_distance, support_distance)
            if min_distance < 0.01:  # 1%範圍內
                score = 60
                direction = '即將突破'
        
        # 檢查突破的持續性
        sustainability = self._check_breakout_sustainability(stock_data, direction)
        if sustainability:
            score += 10
        
        return {
            'confirmed': confirmed,
            'direction': direction,
            'strength': strength,
            'score': score,
            'upward_breakout_pct': upward_breakout_pct if 'upward_breakout_pct' in locals() else 0,
            'downward_breakout_pct': downward_breakout_pct if 'downward_breakout_pct' in locals() else 0,
            'sustainability': sustainability
        }
    
    def _analyze_volume_explosion(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析成交量爆發"""
        
        volume_ratio = stock_data.volume_ratio
        
        confirmed = False
        score = 0
        
        if volume_ratio >= 3.0:
            confirmed = True
            score = 100
        elif volume_ratio >= self.volume_confirmation_ratio:
            confirmed = True
            score = 80
        elif volume_ratio >= 1.5:
            score = 60
        elif volume_ratio >= 1.2:
            score = 40
        else:
            score = 20
        
        # 檢查成交量模式
        volume_pattern = self._analyze_volume_pattern(stock_data.price_history)
        if volume_pattern == 'accumulation':
            score += 15
        elif volume_pattern == 'distribution':
            score -= 10
        
        return {
            'confirmed': confirmed,
            'ratio': volume_ratio,
            'score': min(100, score),
            'volume_pattern': volume_pattern
        }
    
    def _analyze_volatility_pattern(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析波動性模式"""
        
        if len(stock_data.price_history) < 10:
            return {
                'compression_detected': False,
                'expansion_started': False,
                'score': 30
            }
        
        # 計算最近的波動性
        recent_ranges = [day.range_percentage for day in stock_data.price_history[-10:]]
        current_volatility = np.mean(recent_ranges[-3:])  # 最近3天平均
        historical_volatility = np.mean(recent_ranges[:-3])  # 之前7天平均
        
        compression_detected = False
        expansion_started = False
        score = 50
        
        # 檢查波動性壓縮
        if current_volatility < historical_volatility * self.volatility_compression_threshold:
            compression_detected = True
            score = 75
        
        # 檢查波動性擴張
        if current_volatility > historical_volatility * 1.3:
            expansion_started = True
            score = 80
        
        # 檢查ATR模式
        atr_pattern = self._analyze_atr_pattern(stock_data)
        if atr_pattern == 'compression':
            compression_detected = True
            score = max(score, 70)
        elif atr_pattern == 'expansion':
            expansion_started = True
            score = max(score, 75)
        
        return {
            'compression_detected': compression_detected,
            'expansion_started': expansion_started,
            'score': score,
            'current_volatility': current_volatility,
            'historical_volatility': historical_volatility
        }
    
    def _analyze_market_environment_for_breakout(self, stock_data: ComprehensiveStockData, 
                                                market_context: MarketContext) -> float:
        """分析適合突破的市場環境"""
        
        score = 50  # 基礎分數
        
        # 市場波動性環境
        if market_context.volatility_environment == "low_volatility":
            score += 20  # 低波動適合突破
        elif market_context.volatility_environment == "medium_volatility":
            score += 15  # 中等波動也不錯
        else:
            score -= 10  # 高波動增加假突破風險
        
        # 大盤趨勢
        if market_context.is_market_bullish:
            score += 10  # 多頭市場有利向上突破
        
        # 交易時段
        if market_context.trading_session == "opening":
            score += 5   # 開盤時段突破機會多
        elif market_context.trading_session == "closing":
            score -= 15  # 收盤前風險高
        
        return min(100, max(0, score))
    
    def generate_execution_plan(self, stock_data: ComprehensiveStockData, 
                               market_context: MarketContext = None) -> Dict[str, Any]:
        """生成突破交易執行計劃"""
        
        current_price = stock_data.current_price
        atr = stock_data.technical_indicators.atr
        support_resistance = stock_data.calculate_support_resistance()
        
        breakout_analysis = self._analyze_breakout_confirmation(stock_data)
        
        if breakout_analysis['confirmed']:
            if breakout_analysis['direction'] == '向上':
                return self._generate_upward_breakout_plan(stock_data, support_resistance, atr)
            elif breakout_analysis['direction'] == '向下':
                return self._generate_downward_breakout_plan(stock_data, support_resistance, atr)
        else:
            return self._generate_pre_breakout_plan(stock_data, support_resistance, atr)
    
    def _generate_upward_breakout_plan(self, stock_data: ComprehensiveStockData, 
                                      support_resistance: Dict, atr: float) -> Dict[str, Any]:
        """生成向上突破執行計劃"""
        
        current_price = stock_data.current_price
        resistance = support_resistance['resistance']
        
        return {
            'direction': 'long',
            'strategy_type': 'upward_breakout',
            'entry_scenarios': [
                {
                    'scenario': '突破確認',
                    'entry_price': resistance * 1.005,
                    'trigger': '帶量突破壓力位並站穩',
                    'confidence': 85,
                    'wait_for_signals': ['成交量爆發', '突破後回測不破', '技術指標配合']
                },
                {
                    'scenario': '回測進場',
                    'entry_price': resistance * 1.002,
                    'trigger': '突破後回測壓力位變支撐',
                    'confidence': 80,
                    'wait_for_signals': ['回測獲得支撐', '量縮價穩', 'RSI未過熱']
                },
                {
                    'scenario': '追漲進場',
                    'entry_price': current_price + (atr * 0.2),
                    'trigger': '突破後持續強勢上漲',
                    'confidence': 75,
                    'wait_for_signals': ['持續放量', '無明顯阻力', '大盤同步']
                }
            ],
            'stop_loss': resistance * 0.995,  # 跌破突破點即止損
            'profit_targets': [
                {
                    'level': resistance + (resistance - support_resistance['support']) * 1.0,
                    'percentage': 40,
                    'description': '第一目標：等距離測量'
                },
                {
                    'level': resistance + (resistance - support_resistance['support']) * 1.5,
                    'percentage': 35,
                    'description': '第二目標：1.5倍測量'
                },
                {
                    'level': '趨勢跟隨',
                    'percentage': 25,
                    'description': '持續持有直到技術破壞'
                }
            ],
            'risk_reward_ratio': '1:3.5',
            'execution_tips': [
                "突破必須有成交量確認，無量突破多為假突破",
                "進場後設定移動停利保護利潤",
                "注意是否有假突破陷阱",
                "大盤環境要配合個股突破",
                "突破失敗要果斷止損"
            ],
            'risk_management': {
                'max_risk_per_trade': '2%',
                'position_sizing': '根據突破點距離計算',
                'false_breakout_protection': '跌破突破點立即出場',
                'profit_protection': '設定移動停利'
            },
            'warning_signs': [
                "成交量萎縮",
                "出現長上影線",
                "大盤轉弱",
                "跌破突破點"
            ]
        }
    
    def _generate_downward_breakout_plan(self, stock_data: ComprehensiveStockData, 
                                        support_resistance: Dict, atr: float) -> Dict[str, Any]:
        """生成向下突破執行計劃"""
        
        current_price = stock_data.current_price
        support = support_resistance['support']
        
        return {
            'direction': 'short',
            'strategy_type': 'downward_breakout',
            'entry_scenarios': [
                {
                    'scenario': '跌破確認',
                    'entry_price': support * 0.995,
                    'trigger': '帶量跌破支撐位',
                    'confidence': 85,
                    'wait_for_signals': ['成交量放大', '跌破後反彈無力', '技術指標轉弱']
                },
                {
                    'scenario': '反彈空',
                    'entry_price': support * 0.998,
                    'trigger': '跌破後反彈至原支撐位遇阻',
                    'confidence': 80,
                    'wait_for_signals': ['反彈遇阻', '成交量萎縮', '支撐位變壓力位']
                }
            ],
            'stop_loss': support * 1.005,  # 重新站上支撐位即止損
            'profit_targets': [
                {
                    'level': support - (support_resistance['resistance'] - support) * 1.0,
                    'percentage': 50,
                    'description': '第一目標：等距離測量'
                },
                {
                    'level': support - (support_resistance['resistance'] - support) * 1.5,
                    'percentage': 30,
                    'description': '第二目標：1.5倍測量'
                },
                {
                    'level': '趨勢跟隨',
                    'percentage': 20,
                    'description': '持續持有直到技術反轉'
                }
            ],
            'risk_reward_ratio': '1:3.0',
            'execution_tips': [
                "做空風險較高，需要更嚴格的紀律",
                "確認大盤也是弱勢",
                "跌破後的反彈是較好的空點",
                "注意不要在強支撐位盲目放空"
            ]
        }
    
    def _generate_pre_breakout_plan(self, stock_data: ComprehensiveStockData, 
                                   support_resistance: Dict, atr: float) -> Dict[str, Any]:
        """生成突破前等待計劃"""
        
        current_price = stock_data.current_price
        resistance = support_resistance['resistance']
        support = support_resistance['support']
        
        return {
            'direction': 'wait',
            'strategy_type': 'pre_breakout_positioning',
            'recommendation': '等待明確突破信號',
            'reason': '整理型態尚未完成或突破信號不明確',
            'monitoring_levels': [
                {
                    'level': resistance,
                    'action': '準備做多',
                    'condition': '帶量突破並站穩'
                },
                {
                    'level': support,
                    'action': '準備做空',
                    'condition': '帶量跌破並站穩'
                }
            ],
            'pre_positioning_strategy': {
                'approach': '區間操作',
                'buy_zone': f"{support * 1.01:.2f} - {support * 1.02:.2f}",
                'sell_zone': f"{resistance * 0.98:.2f} - {resistance * 0.99:.2f}",
                'stop_loss_buy': support * 0.99,
                'stop_loss_sell': resistance * 1.01
            },
            'breakout_alerts': [
                f"向上突破 {resistance:.2f} 準備追多",
                f"向下跌破 {support:.2f} 準備追空",
                "成交量異常放大時提高警覺",
                "注意假突破陷阱"
            ],
            'wait_for_conditions': [
                "明確的突破信號",
                "成交量配合確認",
                "技術指標同步",
                "大盤環境支持"
            ]
        }
    
    def _detect_rectangle_pattern(self, highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """檢測矩形整理"""
        
        if len(highs) < 5:
            return {'score': 20, 'type': 'rectangle', 'details': 'insufficient_data'}
        
        # 計算高點和低點的一致性
        high_consistency = 1 - (np.std(highs) / np.mean(highs))
        low_consistency = 1 - (np.std(lows) / np.mean(lows))
        
        # 矩形要求高點和低點都相對一致
        if high_consistency > 0.95 and low_consistency > 0.95:
            score = 90
        elif high_consistency > 0.9 and low_consistency > 0.9:
            score = 75
        elif high_consistency > 0.85 and low_consistency > 0.85:
            score = 60
        else:
            score = 30
        
        return {
            'score': score,
            'type': 'rectangle',
            'details': {
                'high_consistency': high_consistency,
                'low_consistency': low_consistency,
                'resistance': np.mean(highs),
                'support': np.mean(lows)
            }
        }
    
    def _detect_triangle_pattern(self, highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """檢測三角形整理"""
        
        if len(highs) < 5:
            return {'score': 20, 'type': 'triangle', 'details': 'insufficient_data'}
        
        # 計算高點和低點的趨勢
        high_trend = self._calculate_trend_slope(highs)
        low_trend = self._calculate_trend_slope(lows)
        
        score = 30
        triangle_type = 'none'
        
        # 上升三角形：高點持平，低點抬升
        if abs(high_trend) < 0.01 and low_trend > 0.01:
            score = 80
            triangle_type = 'ascending'
        
        # 下降三角形：低點持平，高點下降
        elif abs(low_trend) < 0.01 and high_trend < -0.01:
            score = 80
            triangle_type = 'descending'
        
        # 對稱三角形：高點下降，低點上升
        elif high_trend < -0.005 and low_trend > 0.005:
            score = 75
            triangle_type = 'symmetrical'
        
        return {
            'score': score,
            'type': 'triangle',
            'details': {
                'triangle_type': triangle_type,
                'high_trend': high_trend,
                'low_trend': low_trend
            }
        }
    
    def _detect_flag_pattern(self, closes: List[float]) -> Dict[str, Any]:
        """檢測旗形整理"""
        
        if len(closes) < 8:
            return {'score': 20, 'type': 'flag', 'details': 'insufficient_data'}
        
        # 檢查是否有明顯的趨勢後整理
        first_half = closes[:len(closes)//2]
        second_half = closes[len(closes)//2:]
        
        first_trend = self._calculate_trend_slope(first_half)
        second_trend = self._calculate_trend_slope(second_half)
        
        score = 30
        
        # 旗形：先有強趨勢，後有逆向整理
        if abs(first_trend) > 0.02 and abs(second_trend) < 0.01:
            if first_trend * second_trend < 0:  # 方向相反
                score = 70
        
        return {
            'score': score,
            'type': 'flag',
            'details': {
                'first_trend': first_trend,
                'second_trend': second_trend
            }
        }
    
    def _detect_wedge_pattern(self, highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """檢測楔形整理"""
        
        if len(highs) < 5:
            return {'score': 20, 'type': 'wedge', 'details': 'insufficient_data'}
        
        high_trend = self._calculate_trend_slope(highs)
        low_trend = self._calculate_trend_slope(lows)
        
        score = 30
        wedge_type = 'none'
        
        # 上升楔形：高點和低點都上升，但高點斜率較緩
        if high_trend > 0 and low_trend > 0 and low_trend > high_trend:
            score = 70
            wedge_type = 'rising'
        
        # 下降楔形：高點和低點都下降，但低點斜率較急
        elif high_trend < 0 and low_trend < 0 and low_trend < high_trend:
            score = 70
            wedge_type = 'falling'
        
        return {
            'score': score,
            'type': 'wedge',
            'details': {
                'wedge_type': wedge_type,
                'high_trend': high_trend,
                'low_trend': low_trend
            }
        }
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """計算趨勢斜率"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # 線性回歸斜率
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        
        # 正規化為相對斜率
        return slope / np.mean(y) if np.mean(y) != 0 else 0
    
    def _calculate_consolidation_days(self, stock_data: ComprehensiveStockData) -> int:
        """計算整理天數"""
        if len(stock_data.price_history) < 3:
            return 0
        
        # 簡化版本：計算價格在一定範圍內波動的天數
        current_price = stock_data.current_price
        price_range = current_price * 0.05  # 5%範圍
        
        consolidation_days = 0
        for day in reversed(stock_data.price_history):
            if abs(day.close - current_price) <= price_range:
                consolidation_days += 1
            else:
                break
        
        return consolidation_days
    
    def _calculate_price_compression(self, recent_data: List) -> float:
        """計算價格壓縮程度"""
        if len(recent_data) < 5:
            return 0
        
        # 計算最近期間的價格區間
        recent_high = max(day.high for day in recent_data)
        recent_low = min(day.low for day in recent_data)
        recent_range = (recent_high - recent_low) / recent_low
        
        # 計算歷史平均區間
        if len(recent_data) >= 10:
            historical_ranges = []
            for i in range(5, len(recent_data)):
                period_high = max(day.high for day in recent_data[i-5:i])
                period_low = min(day.low for day in recent_data[i-5:i])
                period_range = (period_high - period_low) / period_low
                historical_ranges.append(period_range)
            
            if historical_ranges:
                avg_historical_range = np.mean(historical_ranges)
                compression_ratio = recent_range / avg_historical_range
                return compression_ratio
        
        return 1.0  # 無壓縮
    
    def _check_breakout_sustainability(self, stock_data: ComprehensiveStockData, 
                                      direction: str) -> bool:
        """檢查突破的持續性"""
        
        if direction == 'none':
            return False
        
        # 檢查技術指標是否支持
        indicators = stock_data.technical_indicators
        
        if direction == '向上':
            return (indicators.rsi > 55 and 
                   indicators.macd > 0 and
                   stock_data.volume_ratio > 1.2)
        elif direction == '向下':
            return (indicators.rsi < 45 and 
                   indicators.macd < 0 and
                   stock_data.volume_ratio > 1.2)
        
        return False
    
    def _analyze_volume_pattern(self, price_history: List) -> str:
        """分析成交量模式"""
        if len(price_history) < 5:
            return 'unknown'
        
        recent_volumes = [day.volume for day in price_history[-5:]]
        
        # 檢查是否遞增（累積）
        if all(recent_volumes[i] <= recent_volumes[i+1] for i in range(len(recent_volumes)-1)):
            return 'accumulation'
        
        # 檢查是否遞減（派發）
        if all(recent_volumes[i] >= recent_volumes[i+1] for i in range(len(recent_volumes)-1)):
            return 'distribution'
        
        return 'normal'
    
    def _analyze_atr_pattern(self, stock_data: ComprehensiveStockData) -> str:
        """分析ATR模式"""
        current_atr = stock_data.technical_indicators.atr
        
        if len(stock_data.price_history) < 10:
            return 'unknown'
        
        # 計算歷史ATR
        historical_atr = np.mean([day.range_percentage * stock_data.current_price 
                                 for day in stock_data.price_history[-10:-3]])
        
        if current_atr < historical_atr * 0.7:
            return 'compression'
        elif current_atr > historical_atr * 1.3:
            return 'expansion'
        else:
            return 'normal'
    
    def _identify_false_breakout_risks(self, stock_data: ComprehensiveStockData, 
                                      market_context: MarketContext = None) -> List[str]:
        """識別假突破風險"""
        
        risks = []
        
        # 成交量不足
        if stock_data.volume_ratio < 1.5:
            risks.append("成交量不足，假突破風險高")
        
        # 大盤環境不配合
        if market_context:
            if market_context.volatility_environment == "high_volatility":
                risks.append("高波動環境，假突破頻繁")
            
            current_price = stock_data.current_price
            ma20 = stock_data.technical_indicators.ma20
            
            if market_context.is_market_bullish and current_price < ma20:
                risks.append("大盤多頭但個股弱勢，突破可能無力")
            elif not market_context.is_market_bullish and current_price > ma20:
                risks.append("大盤空頭但個股強勢，突破可能受限")
        
        # 技術指標不配合
        rsi = stock_data.technical_indicators.rsi
        if rsi > 80:
            risks.append("RSI過度超買，向上突破可能乏力")
        elif rsi < 20:
            risks.append("RSI過度超賣，向下突破可能有限")
        
        # 時段風險
        if market_context and market_context.trading_session == "closing":
            risks.append("收盤前突破，隔夜風險高")
        
        return risks
    
    def _assess_breakout_risk_level(self, consolidation_analysis: Dict, 
                                   breakout_analysis: Dict) -> str:
        """評估突破交易風險等級"""
        
        if (consolidation_analysis['pattern_quality'] == 'high' and 
            breakout_analysis['strength'] == 'strong'):
            return '中等'
        elif (consolidation_analysis['has_pattern'] and 
              breakout_analysis['confirmed']):
            return '中高'
        else:
            return '高'  # 突破交易本身就是高風險
    
    def _estimate_breakout_win_rate(self, score: float, breakout_analysis: Dict) -> str:
        """估計突破交易勝率"""
        
        if score >= 85 and breakout_analysis['strength'] == 'strong':
            return '中高'
        elif score >= 75:
            return '中等'
        elif score >= 65:
            return '中等偏低'
        else:
            return '偏低'
    
    def _estimate_breakout_risk_reward(self, consolidation_analysis: Dict, 
                                      breakout_analysis: Dict) -> str:
        """估計突破交易風險回報比"""
        
        if (consolidation_analysis['pattern_quality'] == 'high' and 
            breakout_analysis['strength'] == 'strong'):
            return '極高'  # 高品質突破的回報潛力巨大
        elif consolidation_analysis['has_pattern'] and breakout_analysis['confirmed']:
            return '高'
        else:
            return '中等'