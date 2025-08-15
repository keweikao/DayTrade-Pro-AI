"""
波動性分析引擎
Volatility Analysis Engine - 機會的引擎
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from ..models.stock_data import (
    ComprehensiveStockData, VolatilityAnalysis, StockPrice, MarketContext
)


class VolatilityAnalyzer:
    """波動性分析器 - 基於專業交易員手冊"""
    
    def __init__(self):
        self.thresholds = {
            'excellent_atr_pct': 0.04,      # ATR 4%以上為優秀
            'good_atr_pct': 0.03,           # ATR 3%以上為良好
            'min_atr_pct': 0.02,            # ATR 2%以上為最低標準
            'excellent_range_pct': 0.05,    # 日振幅 5%以上為優秀
            'good_range_pct': 0.03,         # 日振幅 3%以上為良好
            'min_range_pct': 0.02,          # 日振幅 2%以上為最低標準
        }
    
    def assess_volatility(self, stock_data: ComprehensiveStockData, 
                         market_context: MarketContext = None) -> VolatilityAnalysis:
        """
        評估股票波動性
        
        Args:
            stock_data: 完整股票數據
            market_context: 市場環境（可選）
            
        Returns:
            VolatilityAnalysis: 波動性分析結果
        """
        
        # 各項波動性評分
        atr_analysis = self._analyze_atr(stock_data)
        range_analysis = self._analyze_daily_range(stock_data)
        trend_analysis = self._analyze_volatility_trend(stock_data)
        gap_analysis = self._analyze_price_gaps(stock_data)
        intraday_analysis = self._analyze_intraday_patterns(stock_data)
        
        # 計算總分
        total_score = (
            atr_analysis['score'] * 0.30 +         # ATR權重 30%
            range_analysis['score'] * 0.25 +       # 日振幅權重 25%
            trend_analysis['score'] * 0.20 +       # 波動趨勢權重 20%
            gap_analysis['score'] * 0.15 +         # 跳空分析權重 15%
            intraday_analysis['score'] * 0.10      # 盤中模式權重 10%
        )
        
        # 整合詳細說明
        details = []
        details.extend(atr_analysis['details'])
        details.extend(range_analysis['details'])
        details.extend(trend_analysis['details'])
        details.extend(gap_analysis['details'])
        details.extend(intraday_analysis['details'])
        
        # 評級
        grade = self._grade_volatility(total_score)
        opportunity_level = self._assess_opportunity_level(total_score, atr_analysis, range_analysis)
        
        # 波動性趨勢
        volatility_trend = trend_analysis['trend']
        
        # 警告檢查
        warnings = self._generate_volatility_warnings(stock_data, market_context)
        
        return VolatilityAnalysis(
            score=total_score,
            grade=grade,
            details=details,
            warnings=warnings,
            atr_percentage=stock_data.atr_percentage,
            avg_daily_range=stock_data.avg_range_5d,
            opportunity_level=opportunity_level,
            volatility_trend=volatility_trend
        )
    
    def _analyze_atr(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析平均真實波幅 (ATR)"""
        atr_pct = stock_data.atr_percentage
        
        score = 0
        details = []
        
        if atr_pct >= self.thresholds['excellent_atr_pct']:
            score = 100
            details.append(f"極高ATR: {atr_pct:.2%} - 優秀的當沖機會")
        elif atr_pct >= self.thresholds['good_atr_pct']:
            score = 80
            details.append(f"高ATR: {atr_pct:.2%} - 良好的當沖機會")
        elif atr_pct >= self.thresholds['min_atr_pct']:
            score = 60
            details.append(f"中等ATR: {atr_pct:.2%} - 可接受的波動性")
        elif atr_pct >= 0.015:
            score = 40
            details.append(f"較低ATR: {atr_pct:.2%} - 波動性偏低")
        else:
            score = 20
            details.append(f"極低ATR: {atr_pct:.2%} - 波動性不足")
        
        return {'score': score, 'details': details, 'atr_pct': atr_pct}
    
    def _analyze_daily_range(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析日振幅"""
        avg_range = stock_data.avg_range_5d
        
        # 計算近期振幅變化
        if len(stock_data.price_history) >= 10:
            recent_5d = [day.range_percentage for day in stock_data.price_history[-5:]]
            previous_5d = [day.range_percentage for day in stock_data.price_history[-10:-5]]
            
            recent_avg = np.mean(recent_5d)
            previous_avg = np.mean(previous_5d)
            range_change = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
        else:
            range_change = 0
        
        score = 0
        details = []
        
        # 基礎振幅評分
        if avg_range >= self.thresholds['excellent_range_pct']:
            score = 90
            details.append(f"極高日振幅: {avg_range:.2%}")
        elif avg_range >= self.thresholds['good_range_pct']:
            score = 75
            details.append(f"高日振幅: {avg_range:.2%}")
        elif avg_range >= self.thresholds['min_range_pct']:
            score = 60
            details.append(f"中等日振幅: {avg_range:.2%}")
        else:
            score = 40
            details.append(f"低日振幅: {avg_range:.2%}")
        
        # 振幅變化調整
        if range_change > 0.2:  # 振幅增加20%以上
            score += 10
            details.append(f"振幅擴大趨勢: +{range_change:.1%}")
        elif range_change < -0.2:  # 振幅減少20%以上
            score -= 10
            details.append(f"振幅收斂趨勢: {range_change:.1%}")
        
        return {
            'score': min(100, max(0, score)), 
            'details': details, 
            'avg_range': avg_range,
            'range_change': range_change
        }
    
    def _analyze_volatility_trend(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析波動性趨勢"""
        if len(stock_data.price_history) < 20:
            return {'score': 50, 'details': ["歷史數據不足"], 'trend': 'unknown'}
        
        # 計算滾動波動率
        rolling_volatilities = []
        for i in range(10, len(stock_data.price_history)):
            period_data = stock_data.price_history[i-10:i]
            returns = [(day.close / period_data[j-1].close - 1) 
                      for j, day in enumerate(period_data[1:], 1)]
            volatility = np.std(returns) * np.sqrt(252)  # 年化波動率
            rolling_volatilities.append(volatility)
        
        if len(rolling_volatilities) < 5:
            return {'score': 50, 'details': ["波動率計算數據不足"], 'trend': 'unknown'}
        
        # 波動率趨勢分析
        recent_vol = np.mean(rolling_volatilities[-3:])
        earlier_vol = np.mean(rolling_volatilities[-6:-3])
        
        vol_change = (recent_vol - earlier_vol) / earlier_vol if earlier_vol > 0 else 0
        
        score = 50
        details = []
        
        if vol_change > 0.15:  # 波動率增加15%以上
            trend = 'increasing'
            score = 80
            details.append(f"波動率上升趨勢: +{vol_change:.1%}")
        elif vol_change > 0.05:  # 波動率增加5%以上
            trend = 'slightly_increasing'
            score = 70
            details.append(f"波動率微升: +{vol_change:.1%}")
        elif vol_change < -0.15:  # 波動率下降15%以上
            trend = 'decreasing'
            score = 30
            details.append(f"波動率下降趨勢: {vol_change:.1%}")
        elif vol_change < -0.05:  # 波動率下降5%以上
            trend = 'slightly_decreasing'
            score = 40
            details.append(f"波動率微降: {vol_change:.1%}")
        else:
            trend = 'stable'
            score = 60
            details.append("波動率穩定")
        
        # 高波動率穩定也是好的
        if trend == 'stable' and recent_vol > 0.25:  # 年化波動率25%以上
            score = 75
            details.append("高波動率穩定維持")
        
        return {'score': score, 'details': details, 'trend': trend}
    
    def _analyze_price_gaps(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析價格跳空"""
        if len(stock_data.price_history) < 10:
            return {'score': 50, 'details': ["歷史數據不足"]}
        
        # 檢測近期跳空
        gaps = []
        for i in range(1, len(stock_data.price_history)):
            current_day = stock_data.price_history[i]
            previous_day = stock_data.price_history[i-1]
            
            # 向上跳空
            if current_day.low > previous_day.high:
                gap_size = (current_day.low - previous_day.high) / previous_day.high
                gaps.append(('up', gap_size, current_day.date))
            
            # 向下跳空
            elif current_day.high < previous_day.low:
                gap_size = (previous_day.low - current_day.high) / previous_day.low
                gaps.append(('down', gap_size, current_day.date))
        
        # 分析近期跳空（最近10個交易日）
        recent_gaps = [gap for gap in gaps if len(stock_data.price_history) - 
                      next(i for i, day in enumerate(stock_data.price_history) 
                           if day.date == gap[2]) <= 10]
        
        score = 50
        details = []
        
        if len(recent_gaps) == 0:
            details.append("近期無跳空現象")
        else:
            significant_gaps = [gap for gap in recent_gaps if gap[1] > 0.02]  # 2%以上跳空
            
            if len(significant_gaps) > 0:
                score = 80
                details.append(f"近期出現{len(significant_gaps)}次顯著跳空")
                
                # 找出最大跳空
                max_gap = max(significant_gaps, key=lambda x: x[1])
                details.append(f"最大跳空: {max_gap[0]} {max_gap[1]:.2%}")
            else:
                score = 60
                details.append(f"近期出現{len(recent_gaps)}次小幅跳空")
        
        return {'score': score, 'details': details, 'gaps': recent_gaps}
    
    def _analyze_intraday_patterns(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析盤中模式"""
        # 簡化的盤中分析（實際需要分鐘K線數據）
        score = 50
        details = []
        
        # 基於日K線的盤中模式推斷
        if len(stock_data.price_history) >= 5:
            recent_days = stock_data.price_history[-5:]
            
            # 計算開盤vs收盤偏向
            open_close_diffs = [(day.close - day.open) / day.open for day in recent_days]
            avg_oc_diff = np.mean(open_close_diffs)
            
            # 計算高低點模式
            high_low_ratios = [(day.high - day.open) / (day.open - day.low + 0.001) 
                              for day in recent_days if day.open != day.low]
            
            if len(high_low_ratios) > 0:
                avg_hl_ratio = np.mean(high_low_ratios)
                
                if avg_hl_ratio > 1.5:  # 上攻力道強
                    score = 70
                    details.append("近期上攻力道較強")
                elif avg_hl_ratio < 0.7:  # 下殺力道強
                    score = 70
                    details.append("近期下殺力道較強")
                else:
                    score = 60
                    details.append("近期多空拉鋸")
            
            # 振幅一致性
            ranges = [day.range_percentage for day in recent_days]
            range_consistency = 1 - (np.std(ranges) / np.mean(ranges)) if np.mean(ranges) > 0 else 0
            
            if range_consistency > 0.8:
                score += 10
                details.append("振幅模式穩定")
        
        return {'score': score, 'details': details}
    
    def _grade_volatility(self, score: float) -> str:
        """波動性評級"""
        if score >= 85:
            return "極高"
        elif score >= 75:
            return "高"
        elif score >= 65:
            return "中高"
        elif score >= 55:
            return "中等"
        elif score >= 45:
            return "中低"
        elif score >= 35:
            return "低"
        else:
            return "極低"
    
    def _assess_opportunity_level(self, total_score: float, atr_analysis: Dict, 
                                 range_analysis: Dict) -> str:
        """評估機會等級"""
        atr_pct = atr_analysis['atr_pct']
        avg_range = range_analysis['avg_range']
        
        # 綜合ATR和振幅判斷
        if total_score >= 80 and atr_pct >= 0.035 and avg_range >= 0.04:
            return "優秀"
        elif total_score >= 70 and atr_pct >= 0.025 and avg_range >= 0.03:
            return "良好"
        elif total_score >= 60 and atr_pct >= 0.02 and avg_range >= 0.02:
            return "中等"
        elif total_score >= 50:
            return "較低"
        else:
            return "不足"
    
    def _generate_volatility_warnings(self, stock_data: ComprehensiveStockData, 
                                    market_context: MarketContext = None) -> List[str]:
        """生成波動性警告"""
        warnings = []
        
        # ATR警告
        if stock_data.atr_percentage < self.thresholds['min_atr_pct']:
            warnings.append("ATR過低，可能獲利空間不足以覆蓋交易成本")
        
        # 日振幅警告
        if stock_data.avg_range_5d < self.thresholds['min_range_pct']:
            warnings.append("近期振幅過小，當沖機會有限")
        
        # 波動性下降警告
        if len(stock_data.price_history) >= 10:
            recent_range = np.mean([day.range_percentage for day in stock_data.price_history[-5:]])
            earlier_range = np.mean([day.range_percentage for day in stock_data.price_history[-10:-5]])
            
            if recent_range < earlier_range * 0.7:  # 波動性下降30%以上
                warnings.append("波動性呈下降趨勢，交易機會可能減少")
        
        # 市場環境警告
        if market_context:
            if market_context.volatility_environment == "low_volatility":
                warnings.append("整體市場波動性偏低，個股機會可能受限")
            elif market_context.volatility_environment == "high_volatility":
                warnings.append("市場波動性極高，風險控制更為重要")
        
        return warnings
    
    def calculate_expected_range(self, stock_data: ComprehensiveStockData) -> Dict[str, float]:
        """
        計算預期價格區間
        
        Returns:
            Dict: 包含預期高點、低點、振幅等信息
        """
        current_price = stock_data.current_price
        atr = stock_data.technical_indicators.atr
        avg_range_pct = stock_data.avg_range_5d
        
        # 基於ATR的預期區間
        atr_high = current_price + atr
        atr_low = current_price - atr
        
        # 基於歷史振幅的預期區間
        range_high = current_price * (1 + avg_range_pct / 2)
        range_low = current_price * (1 - avg_range_pct / 2)
        
        # 綜合預期區間
        expected_high = (atr_high + range_high) / 2
        expected_low = (atr_low + range_low) / 2
        expected_range_pct = (expected_high - expected_low) / current_price
        
        return {
            'expected_high': expected_high,
            'expected_low': expected_low,
            'expected_range_pct': expected_range_pct,
            'atr_based_high': atr_high,
            'atr_based_low': atr_low,
            'confidence': min(95, max(60, 70 + (avg_range_pct - 0.02) * 500))
        }
    
    def is_suitable_for_scalping(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """
        判斷是否適合超短線交易（剝頭皮）
        
        Returns:
            Dict: 適合度分析
        """
        volatility_analysis = self.assess_volatility(stock_data)
        
        # 超短線需要更高的波動性要求
        if volatility_analysis.score >= 80 and stock_data.atr_percentage >= 0.035:
            suitability = "excellent"
            reason = "極高波動性，非常適合超短線交易"
        elif volatility_analysis.score >= 70 and stock_data.atr_percentage >= 0.025:
            suitability = "good"
            reason = "高波動性，適合超短線交易"
        elif volatility_analysis.score >= 60 and stock_data.atr_percentage >= 0.02:
            suitability = "fair"
            reason = "中等波動性，可考慮超短線交易"
        else:
            suitability = "poor"
            reason = "波動性不足，不適合超短線交易"
        
        expected_range = self.calculate_expected_range(stock_data)
        
        return {
            'suitability': suitability,
            'score': volatility_analysis.score,
            'reason': reason,
            'expected_range': expected_range,
            'minimum_target': current_price * 1.005,  # 最小獲利目標0.5%
            'warnings': volatility_analysis.warnings
        }