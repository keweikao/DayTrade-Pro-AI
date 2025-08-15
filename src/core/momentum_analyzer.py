"""
動能分析引擎
Momentum Analysis Engine - 人氣與催化劑的力量
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from ..models.stock_data import (
    ComprehensiveStockData, MomentumAnalysis, MarketContext, TechnicalIndicators
)


class MomentumAnalyzer:
    """動能分析器 - 基於專業交易員手冊"""
    
    def __init__(self):
        self.momentum_thresholds = {
            'strong_rsi': 60,           # RSI強勢閾值
            'weak_rsi': 40,             # RSI弱勢閾值
            'volume_surge': 2.0,        # 成交量暴增倍數
            'volume_increase': 1.5,     # 成交量增加倍數
            'price_momentum': 0.02,     # 價格動能閾值 2%
            'breakout_threshold': 0.015, # 突破閾值 1.5%
        }
    
    def assess_momentum(self, stock_data: ComprehensiveStockData, 
                       market_context: MarketContext = None) -> MomentumAnalysis:
        """
        評估股票動能
        
        Args:
            stock_data: 完整股票數據
            market_context: 市場環境（可選）
            
        Returns:
            MomentumAnalysis: 動能分析結果
        """
        
        # 各項動能評分
        technical_momentum = self._analyze_technical_momentum(stock_data)
        volume_momentum = self._analyze_volume_momentum(stock_data)
        price_momentum = self._analyze_price_momentum(stock_data)
        breakout_momentum = self._analyze_breakout_momentum(stock_data)
        institutional_momentum = self._analyze_institutional_momentum(stock_data)
        sector_momentum = self._analyze_sector_momentum(stock_data, market_context)
        
        # 計算總分
        total_score = (
            technical_momentum['score'] * 0.25 +       # 技術動能權重 25%
            volume_momentum['score'] * 0.20 +          # 成交量動能權重 20%
            price_momentum['score'] * 0.20 +           # 價格動能權重 20%
            breakout_momentum['score'] * 0.15 +        # 突破動能權重 15%
            institutional_momentum['score'] * 0.10 +   # 法人動能權重 10%
            sector_momentum['score'] * 0.10            # 族群動能權重 10%
        )
        
        # 收集所有催化劑
        catalysts = []
        catalysts.extend(technical_momentum['signals'])
        catalysts.extend(volume_momentum['signals'])
        catalysts.extend(price_momentum['signals'])
        catalysts.extend(breakout_momentum['signals'])
        catalysts.extend(institutional_momentum['signals'])
        catalysts.extend(sector_momentum['signals'])
        
        # 評級
        momentum_grade = self._grade_momentum(total_score)
        
        # 持續性評估
        sustainability = self._assess_sustainability(stock_data, catalysts)
        
        # 警告檢查
        warnings = self._generate_momentum_warnings(stock_data, market_context)
        
        # 整合詳細說明
        details = []
        details.extend(technical_momentum['details'])
        details.extend(volume_momentum['details'])
        details.extend(price_momentum['details'])
        details.extend(breakout_momentum['details'])
        details.extend(institutional_momentum['details'])
        details.extend(sector_momentum['details'])
        
        return MomentumAnalysis(
            score=total_score,
            grade=momentum_grade,
            details=details,
            warnings=warnings,
            catalysts=catalysts,
            momentum_grade=momentum_grade,
            sustainability=sustainability,
            momentum_score=total_score
        )
    
    def _analyze_technical_momentum(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析技術面動能"""
        indicators = stock_data.technical_indicators
        current_price = stock_data.current_price
        
        score = 0
        signals = []
        details = []
        
        # RSI動能分析
        if indicators.rsi > 70:
            score += 20
            signals.append("RSI超買動能")
            details.append(f"RSI達{indicators.rsi:.1f}，動能強勁")
        elif indicators.rsi > self.momentum_thresholds['strong_rsi']:
            score += 15
            signals.append("RSI強勢")
            details.append(f"RSI為{indicators.rsi:.1f}，偏強")
        elif indicators.rsi < 30:
            score += 15
            signals.append("RSI超賣反彈")
            details.append(f"RSI達{indicators.rsi:.1f}，超賣反彈機會")
        
        # MACD動能分析
        if indicators.macd_bullish_crossover:
            score += 25
            signals.append("MACD黃金交叉")
            details.append("MACD快線向上穿越慢線")
        elif indicators.macd > 0 and indicators.macd_histogram > 0:
            score += 15
            signals.append("MACD多頭格局")
            details.append("MACD位於零軸上方且柱狀體為正")
        elif indicators.macd < 0 and indicators.macd_histogram < 0:
            # 空頭動能也是一種動能
            score += 10
            signals.append("MACD空頭格局")
            details.append("MACD位於零軸下方且柱狀體為負")
        
        # KD動能分析
        if indicators.kd_oversold_golden_cross:
            score += 20
            signals.append("KD超賣黃金交叉")
            details.append(f"KD在低檔區({indicators.k:.1f}, {indicators.d:.1f})形成黃金交叉")
        elif indicators.k > 80 and indicators.d > 80 and indicators.k < indicators.d:
            score += 15
            signals.append("KD高檔死叉")
            details.append("KD在高檔區形成死叉")
        
        # 均線動能分析
        ma_trend = stock_data.ma_trend
        if ma_trend == "bullish":
            score += 15
            signals.append("均線多頭排列")
            details.append("短中長期均線呈多頭排列")
        elif ma_trend == "bearish":
            score += 10
            signals.append("均線空頭排列")
            details.append("短中長期均線呈空頭排列")
        
        # 價格相對均線位置
        price_vs_ma20 = stock_data.price_vs_ma20
        if price_vs_ma20 > 0.03:  # 高於MA20 3%以上
            score += 10
            signals.append("價格強勢")
            details.append(f"股價高於20日均線{price_vs_ma20:.1%}")
        elif price_vs_ma20 < -0.03:  # 低於MA20 3%以上
            score += 5
            signals.append("價格弱勢")
            details.append(f"股價低於20日均線{abs(price_vs_ma20):.1%}")
        
        return {'score': min(100, score), 'signals': signals, 'details': details}
    
    def _analyze_volume_momentum(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析成交量動能"""
        volume_ratio = stock_data.volume_ratio
        current_volume = stock_data.current_market.volume
        
        score = 0
        signals = []
        details = []
        
        # 成交量爆發分析
        if volume_ratio >= 3.0:
            score = 100
            signals.append("成交量暴增")
            details.append(f"成交量達平均量{volume_ratio:.1f}倍，動能極強")
        elif volume_ratio >= self.momentum_thresholds['volume_surge']:
            score = 80
            signals.append("成交量大增")
            details.append(f"成交量達平均量{volume_ratio:.1f}倍")
        elif volume_ratio >= self.momentum_thresholds['volume_increase']:
            score = 60
            signals.append("成交量放大")
            details.append(f"成交量達平均量{volume_ratio:.1f}倍")
        elif volume_ratio >= 1.0:
            score = 40
            signals.append("成交量正常")
            details.append(f"成交量達平均量{volume_ratio:.1f}倍")
        else:
            score = 20
            details.append(f"成交量萎縮至平均量{volume_ratio:.1f}倍")
        
        # 成交量趨勢分析
        if len(stock_data.price_history) >= 5:
            recent_volumes = [day.volume for day in stock_data.price_history[-5:]]
            volume_trend = self._calculate_trend(recent_volumes)
            
            if volume_trend > 0.2:  # 成交量上升趨勢
                score += 15
                signals.append("成交量上升趨勢")
                details.append("近期成交量呈上升趨勢")
            elif volume_trend < -0.2:  # 成交量下降趨勢
                score -= 10
                details.append("近期成交量呈下降趨勢")
        
        # 週轉率分析
        turnover_rate = stock_data.turnover_rate
        if turnover_rate > 0.05:  # 週轉率5%以上
            score += 10
            signals.append("高週轉率")
            details.append(f"週轉率達{turnover_rate:.2%}")
        
        return {'score': min(100, max(0, score)), 'signals': signals, 'details': details}
    
    def _analyze_price_momentum(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析價格動能"""
        if len(stock_data.price_history) < 5:
            return {'score': 50, 'signals': [], 'details': ["歷史數據不足"]}
        
        current_price = stock_data.current_price
        recent_prices = [day.close for day in stock_data.price_history[-5:]]
        
        score = 0
        signals = []
        details = []
        
        # 近期價格趨勢
        price_trend = self._calculate_trend(recent_prices)
        
        if price_trend > 0.1:  # 強烈上升趨勢
            score = 90
            signals.append("強勢上漲")
            details.append(f"近期價格強勢上漲，趨勢係數{price_trend:.2f}")
        elif price_trend > 0.05:  # 上升趨勢
            score = 70
            signals.append("溫和上漲")
            details.append(f"近期價格溫和上漲，趨勢係數{price_trend:.2f}")
        elif price_trend < -0.1:  # 強烈下降趨勢
            score = 80  # 下跌動能也是動能
            signals.append("強勢下跌")
            details.append(f"近期價格強勢下跌，趨勢係數{price_trend:.2f}")
        elif price_trend < -0.05:  # 下降趨勢
            score = 60
            signals.append("溫和下跌")
            details.append(f"近期價格溫和下跌，趨勢係數{price_trend:.2f}")
        else:
            score = 40
            details.append("近期價格橫盤整理")
        
        # 價格加速度分析
        if len(recent_prices) >= 4:
            acceleration = self._calculate_acceleration(recent_prices)
            
            if abs(acceleration) > 0.05:  # 加速度明顯
                score += 15
                signals.append("價格加速變化")
                details.append("價格變化出現加速現象")
        
        # 相對強弱分析（相對於自身歷史）
        if len(stock_data.price_history) >= 20:
            recent_20d_high = max(day.high for day in stock_data.price_history[-20:])
            recent_20d_low = min(day.low for day in stock_data.price_history[-20:])
            
            price_position = (current_price - recent_20d_low) / (recent_20d_high - recent_20d_low)
            
            if price_position > 0.8:  # 接近20日高點
                score += 10
                signals.append("接近近期高點")
                details.append(f"價格位於20日區間{price_position:.1%}位置")
            elif price_position < 0.2:  # 接近20日低點
                score += 5
                signals.append("接近近期低點")
                details.append(f"價格位於20日區間{price_position:.1%}位置")
        
        return {'score': min(100, score), 'signals': signals, 'details': details}
    
    def _analyze_breakout_momentum(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析突破動能"""
        current_price = stock_data.current_price
        support_resistance = stock_data.calculate_support_resistance()
        
        score = 0
        signals = []
        details = []
        
        # 突破壓力位分析
        resistance = support_resistance['resistance']
        support = support_resistance['support']
        
        resistance_distance = (current_price - resistance) / resistance
        support_distance = (support - current_price) / support
        
        if resistance_distance > self.momentum_thresholds['breakout_threshold']:
            score = 90
            signals.append("突破壓力位")
            details.append(f"股價突破壓力位{resistance:.2f}元，突破幅度{resistance_distance:.2%}")
        elif resistance_distance > 0:
            score = 70
            signals.append("測試壓力位")
            details.append(f"股價測試壓力位{resistance:.2f}元")
        elif support_distance > self.momentum_thresholds['breakout_threshold']:
            score = 80  # 跌破支撐也是一種動能
            signals.append("跌破支撐位")
            details.append(f"股價跌破支撐位{support:.2f}元，跌破幅度{support_distance:.2%}")
        elif support_distance > 0:
            score = 60
            signals.append("測試支撐位")
            details.append(f"股價測試支撐位{support:.2f}元")
        else:
            score = 50
            details.append("價格在支撐壓力區間內")
        
        # 成交量確認突破
        if stock_data.has_breakout_signal():
            score += 20
            signals.append("量價突破確認")
            details.append("突破伴隨成交量放大確認")
        
        return {'score': min(100, score), 'signals': signals, 'details': details}
    
    def _analyze_institutional_momentum(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析法人動能（模擬）"""
        # 這裡使用模擬數據，實際應用需要接入法人買賣超數據
        score = 50
        signals = []
        details = []
        
        # 基於成交量和價格推斷法人動向
        volume_ratio = stock_data.volume_ratio
        price_vs_ma20 = stock_data.price_vs_ma20
        
        # 大量且價格上漲，可能有法人買超
        if volume_ratio > 2.0 and price_vs_ma20 > 0.02:
            score = 80
            signals.append("推斷法人買超")
            details.append("大量上漲，推斷可能有法人進場")
        elif volume_ratio > 1.5 and price_vs_ma20 > 0:
            score = 70
            signals.append("推斷法人偏多")
            details.append("量增價漲，推斷法人偏多")
        elif volume_ratio > 2.0 and price_vs_ma20 < -0.02:
            score = 75  # 法人賣超也是動能
            signals.append("推斷法人賣超")
            details.append("大量下跌，推斷可能有法人出貨")
        else:
            details.append("法人動向不明顯")
        
        return {'score': score, 'signals': signals, 'details': details}
    
    def _analyze_sector_momentum(self, stock_data: ComprehensiveStockData, 
                                market_context: MarketContext = None) -> Dict[str, Any]:
        """分析族群動能"""
        score = 50
        signals = []
        details = []
        
        if market_context and market_context.sector_performance:
            industry = stock_data.info.industry
            
            # 查找相關產業表現
            sector_performance = market_context.sector_performance.get(industry, 0)
            
            if sector_performance > 0.03:  # 族群漲幅3%以上
                score = 85
                signals.append("領漲族群")
                details.append(f"{industry}族群大漲{sector_performance:.2%}")
            elif sector_performance > 0.01:  # 族群漲幅1%以上
                score = 70
                signals.append("強勢族群")
                details.append(f"{industry}族群上漲{sector_performance:.2%}")
            elif sector_performance < -0.03:  # 族群跌幅3%以上
                score = 75  # 弱勢族群也可能有反彈機會
                signals.append("弱勢族群")
                details.append(f"{industry}族群下跌{abs(sector_performance):.2%}")
            else:
                details.append(f"{industry}族群表現平淡")
        else:
            details.append("族群動向數據不足")
        
        return {'score': score, 'signals': signals, 'details': details}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """計算趨勢係數"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # 簡單線性回歸斜率
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        
        # 正規化到相對變化率
        return slope / np.mean(y) if np.mean(y) != 0 else 0
    
    def _calculate_acceleration(self, values: List[float]) -> float:
        """計算加速度（二階導數）"""
        if len(values) < 3:
            return 0
        
        # 計算一階差分（速度）
        velocity = np.diff(values)
        
        # 計算二階差分（加速度）
        acceleration = np.diff(velocity)
        
        # 返回平均加速度
        return np.mean(acceleration) / np.mean(values) if np.mean(values) != 0 else 0
    
    def _grade_momentum(self, score: float) -> str:
        """動能評級"""
        if score >= 85:
            return "極強"
        elif score >= 75:
            return "強"
        elif score >= 65:
            return "中強"
        elif score >= 55:
            return "中等"
        elif score >= 45:
            return "中弱"
        elif score >= 35:
            return "弱"
        else:
            return "極弱"
    
    def _assess_sustainability(self, stock_data: ComprehensiveStockData, 
                              catalysts: List[str]) -> str:
        """評估動能持續性"""
        
        # 基於催化劑數量和質量評估
        catalyst_count = len(catalysts)
        
        # 檢查是否有基本面支撐
        has_volume_support = stock_data.volume_ratio > 1.5
        has_technical_support = len([c for c in catalysts if "突破" in c or "交叉" in c]) > 0
        has_trend_support = stock_data.ma_trend in ["bullish", "bearish"]
        
        support_count = sum([has_volume_support, has_technical_support, has_trend_support])
        
        if catalyst_count >= 4 and support_count >= 2:
            return "高"
        elif catalyst_count >= 3 and support_count >= 1:
            return "中"
        elif catalyst_count >= 2:
            return "低"
        else:
            return "極低"
    
    def _generate_momentum_warnings(self, stock_data: ComprehensiveStockData, 
                                   market_context: MarketContext = None) -> List[str]:
        """生成動能警告"""
        warnings = []
        
        # 動能不足警告
        momentum_signals = stock_data.get_momentum_signals()
        if len(momentum_signals) < 2:
            warnings.append("動能信號不足，建議等待更明確的信號")
        
        # 成交量不配合警告
        if stock_data.volume_ratio < 1.2:
            warnings.append("成交量未能配合，動能可能不持續")
        
        # 技術指標背離警告
        indicators = stock_data.technical_indicators
        if indicators.rsi > 80:
            warnings.append("RSI過度超買，注意回調風險")
        elif indicators.rsi < 20:
            warnings.append("RSI過度超賣，可能持續弱勢")
        
        # 市場環境警告
        if market_context:
            if not market_context.is_market_bullish and stock_data.price_vs_ma20 > 0.05:
                warnings.append("個股強勢但大盤偏弱，注意系統性風險")
            
            if market_context.volatility_environment == "low_volatility":
                warnings.append("低波動環境，動能可能難以持續")
        
        return warnings
    
    def identify_momentum_pattern(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """
        識別動能模式
        
        Returns:
            Dict: 動能模式分析
        """
        momentum_analysis = self.assess_momentum(stock_data)
        catalysts = momentum_analysis.catalysts
        
        # 動能模式分類
        if "突破" in "".join(catalysts) and "成交量" in "".join(catalysts):
            pattern = "突破動能"
            strength = "強"
            description = "價格突破關鍵位置且有成交量確認"
        elif "RSI" in "".join(catalysts) and "MACD" in "".join(catalysts):
            pattern = "技術動能"
            strength = "中強"
            description = "多個技術指標共振產生動能"
        elif "成交量" in "".join(catalysts):
            pattern = "資金動能"
            strength = "中"
            description = "主要由資金推動的動能"
        elif len(catalysts) >= 3:
            pattern = "複合動能"
            strength = "中強"
            description = "多重因素共同推動的動能"
        else:
            pattern = "弱動能"
            strength = "弱"
            description = "動能信號較少，力度不足"
        
        return {
            'pattern': pattern,
            'strength': strength,
            'description': description,
            'catalyst_count': len(catalysts),
            'sustainability': momentum_analysis.sustainability,
            'recommended_holding_period': self._suggest_holding_period(pattern, strength)
        }
    
    def _suggest_holding_period(self, pattern: str, strength: str) -> str:
        """建議持有期間"""
        if pattern == "突破動能" and strength == "強":
            return "1-3天"
        elif pattern == "技術動能":
            return "半天-1天"
        elif pattern == "資金動能":
            return "1-2天"
        elif pattern == "複合動能":
            return "1-2天"
        else:
            return "數小時內"