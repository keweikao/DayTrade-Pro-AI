"""
逆勢交易策略
Counter Trend Strategy - 均值回歸，捕捉過度延伸後的修正
"""
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass

from ..models.stock_data import ComprehensiveStockData, MarketContext


class CounterTrendStrategy:
    """逆勢交易策略 - 基於專業交易員手冊"""
    
    def __init__(self):
        self.name = "逆勢交易"
        self.philosophy = "均值回歸，漲多必跌，跌深必反"
        self.ideal_conditions = [
            "價格過度延伸",
            "技術指標背離",
            "關鍵支撐/壓力測試",
            "市場情緒極端"
        ]
        
        # 策略參數
        self.oversold_rsi_threshold = 30          # 超賣RSI閾值
        self.overbought_rsi_threshold = 70        # 超買RSI閾值
        self.extreme_rsi_threshold = 20           # 極度超賣/超買閾值
        self.kd_oversold_threshold = 20           # KD超賣閾值
        self.kd_overbought_threshold = 80         # KD超買閾值
        self.price_extension_threshold = 0.08     # 價格過度延伸閾值 8%
        self.volume_divergence_threshold = 1.5    # 量價背離閾值
    
    def evaluate_suitability(self, stock_data: ComprehensiveStockData, 
                            market_context: MarketContext = None) -> Dict[str, Any]:
        """
        評估逆勢策略適用性
        
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
        
        # 1. 過度延伸分析 (權重: 30%)
        extension_analysis = self._analyze_price_extension(stock_data)
        extension_score = extension_analysis['score']
        score += extension_score * 0.30
        
        if extension_analysis['is_overextended']:
            conditions_met.append(f"價格過度延伸: {extension_analysis['detail']}")
            advantages.append(f"過度延伸提供反轉機會: {extension_analysis['detail']}")
            
            if extension_analysis['severity'] == 'extreme':
                conditions_met.append("極度過度延伸")
                advantages.append("極度延伸，反轉動力強")
        else:
            disadvantages.append("價格未過度延伸")
        
        # 2. 技術指標背離分析 (權重: 25%)
        divergence_analysis = self._analyze_divergence_signals(stock_data)
        divergence_score = divergence_analysis['score']
        score += divergence_score * 0.25
        
        for signal in divergence_analysis['divergence_signals']:
            conditions_met.append(f"背離信號: {signal}")
            advantages.append(f"技術背離: {signal}")
        
        if len(divergence_analysis['divergence_signals']) < 1:
            disadvantages.append("缺乏背離信號")
        
        # 3. 關鍵位置測試分析 (權重: 20%)
        key_level_analysis = self._analyze_key_level_test(stock_data)
        key_level_score = key_level_analysis['score']
        score += key_level_score * 0.20
        
        if key_level_analysis['at_key_level']:
            conditions_met.append(f"測試{key_level_analysis['level_type']}: {key_level_analysis['level']:.2f}")
            advantages.append(f"在關鍵{key_level_analysis['level_type']}位置")
            
            if key_level_analysis['bounce_signal']:
                conditions_met.append("出現反彈信號")
                advantages.append("已出現初步反轉信號")
        else:
            disadvantages.append("不在關鍵支撐/壓力位")
        
        # 4. 超買超賣分析 (權重: 15%)
        overbought_oversold_analysis = self._analyze_overbought_oversold(stock_data)
        ob_os_score = overbought_oversold_analysis['score']
        score += ob_os_score * 0.15
        
        for condition in overbought_oversold_analysis['conditions']:
            conditions_met.append(condition)
            advantages.append(f"極端狀態: {condition}")
        
        # 5. 市場情緒分析 (權重: 10%)
        if market_context:
            sentiment_score = self._analyze_market_sentiment(stock_data, market_context)
            score += sentiment_score * 0.10
            
            if market_context.volatility_environment == "high_volatility":
                conditions_met.append("高波動環境")
                advantages.append("高波動有利逆勢操作")
            
            if hasattr(market_context, 'sentiment') and market_context.sentiment in ['extreme_fear', 'extreme_greed']:
                conditions_met.append(f"極端情緒: {market_context.sentiment}")
                advantages.append("極端情緒提供反轉機會")
        
        # 風險等級評估
        risk_level = self._assess_risk_level(extension_analysis, divergence_analysis)
        
        # 勝率與回報預期
        win_rate_expectation = self._estimate_win_rate(score, extension_analysis)
        risk_reward_expectation = self._estimate_risk_reward(extension_analysis, key_level_analysis)
        
        # 檢查逆勢交易的特殊風險
        special_risks = self._identify_special_risks(stock_data, market_context)
        disadvantages.extend(special_risks)
        
        return {
            'score': min(100, max(0, score)),
            'conditions_met': conditions_met,
            'advantages': advantages,
            'disadvantages': disadvantages,
            'risk_level': risk_level,
            'expected_win_rate': win_rate_expectation,
            'expected_risk_reward': risk_reward_expectation,
            'extension_analysis': extension_analysis,
            'divergence_analysis': divergence_analysis,
            'key_level_analysis': key_level_analysis,
            'special_risks': special_risks
        }
    
    def _analyze_price_extension(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析價格過度延伸"""
        
        current_price = stock_data.current_price
        ma20 = stock_data.technical_indicators.ma20
        atr = stock_data.technical_indicators.atr
        
        # 計算價格相對均線的偏離程度
        price_deviation = abs(current_price - ma20) / ma20
        
        # 計算價格相對ATR的偏離程度
        atr_deviation = abs(current_price - ma20) / atr if atr > 0 else 0
        
        # 判斷過度延伸
        is_overextended = False
        severity = 'none'
        detail = ""
        score = 30  # 基礎分數
        
        if price_deviation > self.price_extension_threshold:
            is_overextended = True
            if current_price > ma20:
                detail = f"超漲{price_deviation:.2%}"
            else:
                detail = f"超跌{price_deviation:.2%}"
            
            if price_deviation > 0.12:  # 12%以上偏離
                severity = 'extreme'
                score = 90
            elif price_deviation > self.price_extension_threshold:
                severity = 'moderate'
                score = 70
        
        # ATR偏離度加分
        if atr_deviation > 2.5:  # 超過2.5倍ATR
            score += 15
            if not detail:
                detail = f"偏離均線{atr_deviation:.1f}倍ATR"
        
        # 檢查拋物線走勢
        if self._detect_parabolic_move(stock_data):
            score += 20
            if detail:
                detail += "，呈拋物線走勢"
            else:
                detail = "拋物線走勢"
            is_overextended = True
            severity = 'extreme'
        
        return {
            'is_overextended': is_overextended,
            'severity': severity,
            'detail': detail,
            'score': min(100, score),
            'price_deviation': price_deviation,
            'atr_deviation': atr_deviation
        }
    
    def _analyze_divergence_signals(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析技術指標背離"""
        
        indicators = stock_data.technical_indicators
        divergence_signals = []
        score = 0
        
        # RSI背離分析
        rsi_divergence = self._check_rsi_divergence(stock_data)
        if rsi_divergence:
            divergence_signals.append(f"RSI{rsi_divergence}")
            score += 35
        
        # MACD背離分析
        macd_divergence = self._check_macd_divergence(stock_data)
        if macd_divergence:
            divergence_signals.append(f"MACD{macd_divergence}")
            score += 30
        
        # KD背離分析
        kd_divergence = self._check_kd_divergence(stock_data)
        if kd_divergence:
            divergence_signals.append(f"KD{kd_divergence}")
            score += 25
        
        # 成交量背離
        volume_divergence = self._check_volume_divergence(stock_data)
        if volume_divergence:
            divergence_signals.append(f"量價背離: {volume_divergence}")
            score += 20
        
        return {
            'score': min(100, score),
            'divergence_signals': divergence_signals,
            'divergence_strength': 'strong' if score > 70 else 'moderate' if score > 40 else 'weak'
        }
    
    def _analyze_key_level_test(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析關鍵位置測試"""
        
        current_price = stock_data.current_price
        support_resistance = stock_data.calculate_support_resistance()
        
        support = support_resistance['support']
        resistance = support_resistance['resistance']
        
        at_key_level = False
        level_type = ""
        level = 0
        bounce_signal = False
        score = 30  # 基礎分數
        
        # 檢查是否在關鍵支撐位
        support_distance = abs(current_price - support) / support
        if support_distance < 0.02:  # 2%範圍內
            at_key_level = True
            level_type = "支撐位"
            level = support
            score = 70
            
            # 檢查是否有反彈信號
            if self._check_bounce_signals_at_support(stock_data):
                bounce_signal = True
                score = 85
        
        # 檢查是否在關鍵壓力位
        resistance_distance = abs(current_price - resistance) / resistance
        if resistance_distance < 0.02:  # 2%範圍內
            at_key_level = True
            level_type = "壓力位"
            level = resistance
            score = 70
            
            # 檢查是否有回落信號
            if self._check_rejection_signals_at_resistance(stock_data):
                bounce_signal = True
                score = 85
        
        # 心理關卡檢查
        psychological_level = self._check_psychological_levels(current_price)
        if psychological_level:
            if not at_key_level:  # 避免重複計分
                at_key_level = True
                level_type = "心理關卡"
                level = psychological_level
                score = 60
        
        return {
            'at_key_level': at_key_level,
            'level_type': level_type,
            'level': level,
            'bounce_signal': bounce_signal,
            'score': score
        }
    
    def _analyze_overbought_oversold(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析超買超賣狀態"""
        
        indicators = stock_data.technical_indicators
        conditions = []
        score = 0
        
        # RSI超買超賣
        if indicators.rsi <= self.extreme_rsi_threshold:
            conditions.append(f"RSI極度超賣({indicators.rsi:.1f})")
            score += 40
        elif indicators.rsi <= self.oversold_rsi_threshold:
            conditions.append(f"RSI超賣({indicators.rsi:.1f})")
            score += 30
        elif indicators.rsi >= (100 - self.extreme_rsi_threshold):
            conditions.append(f"RSI極度超買({indicators.rsi:.1f})")
            score += 40
        elif indicators.rsi >= self.overbought_rsi_threshold:
            conditions.append(f"RSI超買({indicators.rsi:.1f})")
            score += 30
        
        # KD超買超賣
        if indicators.k <= self.kd_oversold_threshold and indicators.d <= self.kd_oversold_threshold:
            conditions.append(f"KD超賣區({indicators.k:.1f}, {indicators.d:.1f})")
            score += 25
        elif indicators.k >= self.kd_overbought_threshold and indicators.d >= self.kd_overbought_threshold:
            conditions.append(f"KD超買區({indicators.k:.1f}, {indicators.d:.1f})")
            score += 25
        
        # 多重指標確認
        if len(conditions) >= 2:
            score += 15  # 多重確認加分
        
        return {
            'score': min(100, score),
            'conditions': conditions,
            'extreme_level': 'high' if score > 60 else 'moderate' if score > 30 else 'low'
        }
    
    def _analyze_market_sentiment(self, stock_data: ComprehensiveStockData, 
                                 market_context: MarketContext) -> float:
        """分析市場情緒"""
        
        score = 50  # 基礎分數
        
        # VIX恐慌指數
        if hasattr(market_context, 'vix'):
            vix = market_context.vix
            if vix > 30:  # 高恐慌
                score += 25
            elif vix < 15:  # 低恐慌（可能過度樂觀）
                score += 15
        
        # 市場波動性環境
        if market_context.volatility_environment == "high_volatility":
            score += 15  # 高波動有利逆勢
        elif market_context.volatility_environment == "low_volatility":
            score -= 10  # 低波動不利逆勢
        
        # 外資動向（如果有數據）
        if hasattr(market_context, 'foreign_investment'):
            foreign_flow = market_context.foreign_investment
            if abs(foreign_flow) > 100:  # 外資大進大出
                score += 10
        
        return min(100, max(0, score))
    
    def generate_execution_plan(self, stock_data: ComprehensiveStockData, 
                               market_context: MarketContext = None) -> Dict[str, Any]:
        """生成逆勢交易執行計劃"""
        
        current_price = stock_data.current_price
        atr = stock_data.technical_indicators.atr
        indicators = stock_data.technical_indicators
        
        # 確定逆勢方向
        extension_analysis = self._analyze_price_extension(stock_data)
        overbought_oversold = self._analyze_overbought_oversold(stock_data)
        
        # 判斷應該做多還是做空
        if (indicators.rsi <= self.oversold_rsi_threshold or 
            current_price < stock_data.technical_indicators.ma20 * 0.95):
            return self._generate_counter_trend_long_plan(stock_data, atr)
        elif (indicators.rsi >= self.overbought_rsi_threshold or 
              current_price > stock_data.technical_indicators.ma20 * 1.05):
            return self._generate_counter_trend_short_plan(stock_data, atr)
        else:
            return self._generate_wait_plan(stock_data)
    
    def _generate_counter_trend_long_plan(self, stock_data: ComprehensiveStockData, 
                                         atr: float) -> Dict[str, Any]:
        """生成逆勢做多計劃"""
        
        current_price = stock_data.current_price
        support_resistance = stock_data.calculate_support_resistance()
        support = support_resistance['support']
        
        return {
            'direction': 'long',
            'strategy_type': 'counter_trend_reversal',
            'entry_scenarios': [
                {
                    'scenario': '超賣反彈',
                    'entry_price': current_price,
                    'trigger': 'RSI超賣且出現反彈信號',
                    'confidence': 70,
                    'wait_for_signals': ['RSI背離確認', '支撐位止跌', '量縮價穩']
                },
                {
                    'scenario': '支撐位反彈',
                    'entry_price': support * 1.005,
                    'trigger': '價格測試支撐位後反彈',
                    'confidence': 75,
                    'wait_for_signals': ['支撐位獲得確認', '成交量萎縮', '技術指標鈍化']
                },
                {
                    'scenario': '背離確認',
                    'entry_price': current_price * 0.998,
                    'trigger': '價格創新低但指標背離',
                    'confidence': 80,
                    'wait_for_signals': ['多重指標背離', '量價背離', '反轉K線確認']
                }
            ],
            'stop_loss': min(support * 0.99, current_price - (atr * 1.2)),
            'profit_targets': [
                {
                    'level': current_price + (atr * 1.5),
                    'percentage': 60,
                    'description': '第一目標：快速獲利了結'
                },
                {
                    'level': stock_data.technical_indicators.ma20,
                    'percentage': 30,
                    'description': '回歸均線'
                },
                {
                    'level': current_price + (atr * 2.5),
                    'percentage': 10,
                    'description': '超額利潤'
                }
            ],
            'risk_reward_ratio': '1:1.8',
            'execution_tips': [
                "逆勢操作風險較高，嚴格執行停損",
                "分批進場，降低平均成本",
                "快進快出，不要貪心",
                "密切關注技術指標的背離信號",
                "一旦趨勢重新確立，立即止損"
            ],
            'risk_management': {
                'max_risk_per_trade': '1.5%',
                'position_sizing': '小倉位測試',
                'holding_period': '通常1-3天',
                'exit_discipline': '嚴格按計劃執行'
            },
            'warning_signs': [
                "成交量未能萎縮",
                "技術指標未出現背離",
                "跌破重要支撑位",
                "大盤環境惡化"
            ]
        }
    
    def _generate_counter_trend_short_plan(self, stock_data: ComprehensiveStockData, 
                                          atr: float) -> Dict[str, Any]:
        """生成逆勢做空計劃"""
        
        current_price = stock_data.current_price
        support_resistance = stock_data.calculate_support_resistance()
        resistance = support_resistance['resistance']
        
        return {
            'direction': 'short',
            'strategy_type': 'counter_trend_reversal',
            'entry_scenarios': [
                {
                    'scenario': '超買回落',
                    'entry_price': current_price,
                    'trigger': 'RSI超買且出現回落信號',
                    'confidence': 70,
                    'wait_for_signals': ['RSI背離確認', '壓力位遇阻', '量增價滯']
                },
                {
                    'scenario': '壓力位遇阻',
                    'entry_price': resistance * 0.995,
                    'trigger': '價格測試壓力位後回落',
                    'confidence': 75,
                    'wait_for_signals': ['壓力位確認有效', '上影線出現', '成交量放大但收黑']
                }
            ],
            'stop_loss': max(resistance * 1.01, current_price + (atr * 1.2)),
            'profit_targets': [
                {
                    'level': current_price - (atr * 1.5),
                    'percentage': 60,
                    'description': '第一目標：快速獲利了結'
                },
                {
                    'level': stock_data.technical_indicators.ma20,
                    'percentage': 30,
                    'description': '回歸均線'
                },
                {
                    'level': current_price - (atr * 2.5),
                    'percentage': 10,
                    'description': '超額利潤'
                }
            ],
            'risk_reward_ratio': '1:1.8',
            'execution_tips': [
                "做空風險更高，更需要嚴格紀律",
                "注意不要在強勢趨勢中逆勢操作",
                "快速獲利了結，避免貪心",
                "密切關注大盤走勢"
            ]
        }
    
    def _generate_wait_plan(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """生成等待計劃"""
        
        return {
            'direction': 'wait',
            'recommendation': '等待更好的逆勢機會',
            'reason': '當前未達到逆勢操作的理想條件',
            'wait_for_conditions': [
                'RSI進入超買或超賣區域',
                '出現明顯的技術指標背離',
                '價格測試關鍵支撐或壓力位',
                '市場情緒達到極端狀態'
            ],
            'monitoring_indicators': [
                f"RSI: {stock_data.technical_indicators.rsi:.1f} (等待<30或>70)",
                f"價格vs MA20: {stock_data.price_vs_ma20:.2%} (等待>5%或<-5%)",
                "關注MACD和KD的背離信號"
            ]
        }
    
    def _detect_parabolic_move(self, stock_data: ComprehensiveStockData) -> bool:
        """檢測拋物線走勢"""
        if len(stock_data.price_history) < 5:
            return False
        
        recent_closes = [day.close for day in stock_data.price_history[-5:]]
        
        # 檢查是否連續上漲且漲幅遞增
        consecutive_gains = 0
        acceleration = True
        
        for i in range(1, len(recent_closes)):
            if recent_closes[i] > recent_closes[i-1]:
                consecutive_gains += 1
                if i >= 2:
                    prev_gain = (recent_closes[i-1] - recent_closes[i-2]) / recent_closes[i-2]
                    curr_gain = (recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1]
                    if curr_gain <= prev_gain:
                        acceleration = False
            else:
                break
        
        return consecutive_gains >= 3 and acceleration
    
    def _check_rsi_divergence(self, stock_data: ComprehensiveStockData) -> Optional[str]:
        """檢查RSI背離"""
        # 簡化版本：實際需要歷史RSI數據
        current_rsi = stock_data.technical_indicators.rsi
        current_price = stock_data.current_price
        
        if len(stock_data.price_history) >= 10:
            # 檢查價格新高但RSI未創新高（看跌背離）
            recent_high = max(day.high for day in stock_data.price_history[-10:])
            if current_price >= recent_high * 0.99 and current_rsi < 60:
                return "看跌背離"
            
            # 檢查價格新低但RSI未創新低（看漲背離）
            recent_low = min(day.low for day in stock_data.price_history[-10:])
            if current_price <= recent_low * 1.01 and current_rsi > 40:
                return "看漲背離"
        
        return None
    
    def _check_macd_divergence(self, stock_data: ComprehensiveStockData) -> Optional[str]:
        """檢查MACD背離"""
        # 簡化版本：基於當前MACD狀態推斷
        macd = stock_data.technical_indicators.macd
        macd_histogram = stock_data.technical_indicators.macd_histogram
        
        # 如果MACD柱狀體與價格走勢相反
        price_vs_ma20 = stock_data.price_vs_ma20
        
        if price_vs_ma20 > 0.02 and macd_histogram < 0:
            return "MACD看跌背離"
        elif price_vs_ma20 < -0.02 and macd_histogram > 0:
            return "MACD看漲背離"
        
        return None
    
    def _check_kd_divergence(self, stock_data: ComprehensiveStockData) -> Optional[str]:
        """檢查KD背離"""
        # 簡化版本
        k = stock_data.technical_indicators.k
        d = stock_data.technical_indicators.d
        current_price = stock_data.current_price
        
        # 基於KD值與價格位置的關係判斷
        if current_price > stock_data.technical_indicators.ma20 * 1.03 and k < 50:
            return "KD看跌背離"
        elif current_price < stock_data.technical_indicators.ma20 * 0.97 and k > 50:
            return "KD看漲背離"
        
        return None
    
    def _check_volume_divergence(self, stock_data: ComprehensiveStockData) -> Optional[str]:
        """檢查量價背離"""
        volume_ratio = stock_data.volume_ratio
        price_change = abs(stock_data.price_vs_ma20)
        
        # 價漲量縮
        if stock_data.price_vs_ma20 > 0.02 and volume_ratio < 0.8:
            return "價漲量縮"
        # 價跌量增
        elif stock_data.price_vs_ma20 < -0.02 and volume_ratio > self.volume_divergence_threshold:
            return "價跌量增"
        
        return None
    
    def _check_bounce_signals_at_support(self, stock_data: ComprehensiveStockData) -> bool:
        """檢查支撐位反彈信號"""
        # 簡化版本：檢查是否有反彈的基本條件
        volume_ratio = stock_data.volume_ratio
        rsi = stock_data.technical_indicators.rsi
        
        # 量縮且RSI超賣
        return volume_ratio < 1.0 and rsi < 35
    
    def _check_rejection_signals_at_resistance(self, stock_data: ComprehensiveStockData) -> bool:
        """檢查壓力位遇阻信號"""
        # 簡化版本：檢查是否有遇阻的基本條件
        volume_ratio = stock_data.volume_ratio
        rsi = stock_data.technical_indicators.rsi
        
        # 量增價滯且RSI超買
        return volume_ratio > 1.3 and rsi > 65
    
    def _check_psychological_levels(self, price: float) -> Optional[float]:
        """檢查心理關卡"""
        # 檢查整數關卡
        if price >= 100:
            rounded_hundreds = round(price / 100) * 100
            if abs(price - rounded_hundreds) / price < 0.02:
                return rounded_hundreds
        
        # 檢查50元關卡
        rounded_fifty = round(price / 50) * 50
        if abs(price - rounded_fifty) / price < 0.03:
            return rounded_fifty
        
        return None
    
    def _identify_special_risks(self, stock_data: ComprehensiveStockData, 
                               market_context: MarketContext = None) -> List[str]:
        """識別逆勢交易的特殊風險"""
        
        risks = []
        
        # 趨勢風險
        if stock_data.ma_trend in ['bullish', 'bearish']:
            risks.append("與主要趨勢相逆，風險較高")
        
        # 動能風險
        momentum_signals = stock_data.get_momentum_signals()
        if len(momentum_signals) >= 3:
            risks.append("動能信號強勁，逆勢風險大")
        
        # 成交量風險
        if stock_data.volume_ratio > 2.0:
            risks.append("成交量暴增，趨勢可能持續")
        
        # 市場環境風險
        if market_context and market_context.volatility_environment == "low_volatility":
            risks.append("低波動環境，反轉動力不足")
        
        # 時間風險
        if market_context and market_context.trading_session == "closing":
            risks.append("收盤前逆勢操作風險極高")
        
        return risks
    
    def _assess_risk_level(self, extension_analysis: Dict, divergence_analysis: Dict) -> str:
        """評估風險等級"""
        
        if (extension_analysis['severity'] == 'extreme' and 
            divergence_analysis['divergence_strength'] == 'strong'):
            return '中等'  # 雖然逆勢但有強烈信號支撐
        elif extension_analysis['is_overextended'] and len(divergence_analysis['divergence_signals']) > 0:
            return '中高'
        else:
            return '高'  # 逆勢操作本身就是高風險
    
    def _estimate_win_rate(self, score: float, extension_analysis: Dict) -> str:
        """估計勝率"""
        
        if score >= 80 and extension_analysis['severity'] == 'extreme':
            return '中高'
        elif score >= 70:
            return '中等'
        elif score >= 60:
            return '中等偏低'
        else:
            return '偏低'
    
    def _estimate_risk_reward(self, extension_analysis: Dict, key_level_analysis: Dict) -> str:
        """估計風險回報比"""
        
        if (extension_analysis['severity'] == 'extreme' and 
            key_level_analysis['at_key_level']):
            return '中高'  # 極端位置的反轉機會
        elif extension_analysis['is_overextended']:
            return '中等'
        else:
            return '偏低'