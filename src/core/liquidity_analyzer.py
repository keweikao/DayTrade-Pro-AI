"""
流動性分析引擎
Liquidity Analysis Engine - 當沖交易的命脈
"""
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from ..models.stock_data import (
    ComprehensiveStockData, LiquidityAnalysis, MarketContext
)


class LiquidityAnalyzer:
    """流動性分析器 - 基於專業交易員手冊"""
    
    def __init__(self):
        self.criteria = {
            'excellent_volume_threshold': 50000,    # 5萬張為優秀
            'good_volume_threshold': 20000,         # 2萬張為良好
            'min_volume_threshold': 10000,          # 1萬張為最低標準
            'max_spread_percentage': 0.002,         # 最大價差 0.2%
            'min_turnover_rate': 0.02,              # 最小週轉率 2%
            'min_market_depth': 500,                # 最小五檔深度 500張
        }
    
    def assess_liquidity(self, stock_data: ComprehensiveStockData, 
                        market_context: MarketContext = None) -> LiquidityAnalysis:
        """
        評估股票流動性
        
        Args:
            stock_data: 完整股票數據
            market_context: 市場環境（可選）
            
        Returns:
            LiquidityAnalysis: 流動性分析結果
        """
        
        # 各項流動性評分
        volume_analysis = self._analyze_volume(stock_data)
        spread_analysis = self._analyze_spread(stock_data)
        depth_analysis = self._analyze_market_depth(stock_data)
        turnover_analysis = self._analyze_turnover(stock_data)
        consistency_analysis = self._analyze_volume_consistency(stock_data)
        
        # 計算總分
        total_score = (
            volume_analysis['score'] * 0.35 +      # 成交量權重 35%
            spread_analysis['score'] * 0.25 +      # 價差權重 25%
            turnover_analysis['score'] * 0.20 +    # 週轉率權重 20%
            depth_analysis['score'] * 0.15 +       # 市場深度權重 15%
            consistency_analysis['score'] * 0.05   # 一致性權重 5%
        )
        
        # 整合詳細說明
        details = []
        details.extend(volume_analysis['details'])
        details.extend(spread_analysis['details'])
        details.extend(turnover_analysis['details'])
        details.extend(depth_analysis['details'])
        details.extend(consistency_analysis['details'])
        
        # 評級
        grade = self._grade_liquidity(total_score)
        
        # 可交易性判斷
        is_tradeable = self._determine_tradeability(
            volume_analysis, spread_analysis, turnover_analysis
        )
        
        # 警告檢查
        warnings = self._generate_liquidity_warnings(stock_data, market_context)
        
        return LiquidityAnalysis(
            score=total_score,
            grade=grade,
            details=details,
            warnings=warnings,
            is_tradeable=is_tradeable,
            volume_score=volume_analysis['score'],
            spread_score=spread_analysis['score'],
            depth_score=depth_analysis['score']
        )
    
    def _analyze_volume(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析成交量"""
        avg_volume = stock_data.avg_volume_20d
        current_volume = stock_data.current_market.volume
        volume_ratio = stock_data.volume_ratio
        
        score = 0
        details = []
        
        # 平均成交量評分
        if avg_volume >= self.criteria['excellent_volume_threshold']:
            score += 40
            details.append(f"超高平均成交量: {avg_volume:,.0f}張/日")
        elif avg_volume >= self.criteria['good_volume_threshold']:
            score += 30
            details.append(f"高平均成交量: {avg_volume:,.0f}張/日")
        elif avg_volume >= self.criteria['min_volume_threshold']:
            score += 20
            details.append(f"中等成交量: {avg_volume:,.0f}張/日")
        else:
            details.append(f"成交量不足: {avg_volume:,.0f}張/日")
        
        # 當日成交量比率評分
        if volume_ratio >= 2.0:
            score += 20
            details.append(f"當日量能爆發: {volume_ratio:.1f}倍")
        elif volume_ratio >= 1.5:
            score += 15
            details.append(f"當日量能放大: {volume_ratio:.1f}倍")
        elif volume_ratio >= 1.0:
            score += 10
            details.append(f"當日量能正常: {volume_ratio:.1f}倍")
        else:
            details.append(f"當日量能萎縮: {volume_ratio:.1f}倍")
        
        return {'score': min(100, score), 'details': details}
    
    def _analyze_spread(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析買賣價差"""
        spread_pct = stock_data.current_market.spread_percentage
        
        score = 0
        details = []
        
        if spread_pct <= 0.001:  # 0.1%以下
            score = 100
            details.append(f"極窄價差: {spread_pct:.3%}")
        elif spread_pct <= 0.002:  # 0.2%以下
            score = 80
            details.append(f"窄價差: {spread_pct:.3%}")
        elif spread_pct <= 0.005:  # 0.5%以下
            score = 60
            details.append(f"中等價差: {spread_pct:.3%}")
        elif spread_pct <= 0.01:   # 1%以下
            score = 40
            details.append(f"較寬價差: {spread_pct:.3%}")
        else:
            score = 20
            details.append(f"過寬價差: {spread_pct:.3%}")
        
        return {'score': score, 'details': details}
    
    def _analyze_turnover(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析週轉率"""
        turnover_rate = stock_data.turnover_rate
        
        score = 0
        details = []
        
        if turnover_rate >= 0.05:  # 5%以上
            score = 100
            details.append(f"超高週轉率: {turnover_rate:.2%}")
        elif turnover_rate >= 0.03:  # 3%以上
            score = 80
            details.append(f"高週轉率: {turnover_rate:.2%}")
        elif turnover_rate >= 0.02:  # 2%以上
            score = 60
            details.append(f"中等週轉率: {turnover_rate:.2%}")
        elif turnover_rate >= 0.01:  # 1%以上
            score = 40
            details.append(f"較低週轉率: {turnover_rate:.2%}")
        else:
            score = 20
            details.append(f"極低週轉率: {turnover_rate:.2%}")
        
        return {'score': score, 'details': details}
    
    def _analyze_market_depth(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析市場深度"""
        # 簡化的市場深度分析（實際需要五檔數據）
        market_data = stock_data.current_market
        estimated_depth = market_data.bid_size + market_data.ask_size
        
        score = 0
        details = []
        
        if estimated_depth >= 1000:
            score = 100
            details.append(f"市場深度充足: {estimated_depth:,}張")
        elif estimated_depth >= 500:
            score = 80
            details.append(f"市場深度良好: {estimated_depth:,}張")
        elif estimated_depth >= 200:
            score = 60
            details.append(f"市場深度中等: {estimated_depth:,}張")
        else:
            score = 40
            details.append(f"市場深度較淺: {estimated_depth:,}張")
        
        return {'score': score, 'details': details}
    
    def _analyze_volume_consistency(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """分析成交量一致性"""
        if len(stock_data.price_history) < 5:
            return {'score': 50, 'details': ["歷史數據不足"]}
        
        recent_volumes = [day.volume for day in stock_data.price_history[-5:]]
        volume_cv = np.std(recent_volumes) / np.mean(recent_volumes)  # 變異係數
        
        score = 0
        details = []
        
        if volume_cv <= 0.3:  # 變異係數低，代表一致性高
            score = 100
            details.append("成交量穩定一致")
        elif volume_cv <= 0.5:
            score = 80
            details.append("成交量相對穩定")
        elif volume_cv <= 0.8:
            score = 60
            details.append("成交量波動中等")
        else:
            score = 40
            details.append("成交量波動較大")
        
        return {'score': score, 'details': details}
    
    def _grade_liquidity(self, score: float) -> str:
        """流動性評級"""
        if score >= 85:
            return "A+"
        elif score >= 75:
            return "A"
        elif score >= 65:
            return "B+"
        elif score >= 55:
            return "B"
        elif score >= 45:
            return "C+"
        elif score >= 35:
            return "C"
        else:
            return "D"
    
    def _determine_tradeability(self, volume_analysis: Dict, spread_analysis: Dict, 
                               turnover_analysis: Dict) -> bool:
        """判斷是否適合交易"""
        # 核心標準：成交量、價差、週轉率都要達到最低標準
        min_volume_ok = volume_analysis['score'] >= 20
        max_spread_ok = spread_analysis['score'] >= 60
        min_turnover_ok = turnover_analysis['score'] >= 40
        
        return min_volume_ok and max_spread_ok and min_turnover_ok
    
    def _generate_liquidity_warnings(self, stock_data: ComprehensiveStockData, 
                                   market_context: MarketContext = None) -> List[str]:
        """生成流動性警告"""
        warnings = []
        
        # 成交量警告
        if stock_data.avg_volume_20d < self.criteria['min_volume_threshold']:
            warnings.append("平均成交量低於建議標準，可能面臨流動性風險")
        
        # 價差警告
        if stock_data.current_market.spread_percentage > self.criteria['max_spread_percentage']:
            warnings.append("買賣價差過大，交易成本較高")
        
        # 週轉率警告
        if stock_data.turnover_rate < self.criteria['min_turnover_rate']:
            warnings.append("週轉率偏低，市場活躍度不足")
        
        # 當日量能警告
        if stock_data.volume_ratio < 0.5:
            warnings.append("當日成交量萎縮，可能不利於當沖操作")
        
        # 市場環境警告
        if market_context and market_context.trading_session == "closing":
            warnings.append("接近收盤時間，流動性可能下降")
        
        return warnings
    
    def get_optimal_trade_size(self, stock_data: ComprehensiveStockData, 
                              target_impact: float = 0.001) -> int:
        """
        計算最佳交易規模（避免市場衝擊）
        
        Args:
            stock_data: 股票數據
            target_impact: 目標市場衝擊（預設0.1%）
            
        Returns:
            int: 建議最大交易張數
        """
        
        # 簡化的市場衝擊模型
        avg_volume = stock_data.avg_volume_20d
        
        # 假設單日成交量的1%不會造成顯著衝擊
        max_daily_impact_volume = avg_volume * 0.01
        
        # 轉換為張數
        max_lots = int(max_daily_impact_volume)
        
        # 考慮流動性等級調整
        liquidity_analysis = self.assess_liquidity(stock_data)
        
        if liquidity_analysis.grade in ['A+', 'A']:
            multiplier = 1.5
        elif liquidity_analysis.grade in ['B+', 'B']:
            multiplier = 1.0
        else:
            multiplier = 0.5
        
        return int(max_lots * multiplier)
    
    def is_suitable_for_daytrading(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """
        綜合判斷是否適合當沖
        
        Returns:
            Dict: 包含適合度、原因、建議等信息
        """
        
        liquidity_analysis = self.assess_liquidity(stock_data)
        
        # 當沖適合度判斷
        if liquidity_analysis.score >= 75 and liquidity_analysis.is_tradeable:
            suitability = "excellent"
            reason = "流動性優秀，非常適合當沖交易"
        elif liquidity_analysis.score >= 60 and liquidity_analysis.is_tradeable:
            suitability = "good"
            reason = "流動性良好，適合當沖交易"
        elif liquidity_analysis.score >= 45:
            suitability = "fair"
            reason = "流動性中等，可考慮小額當沖"
        else:
            suitability = "poor"
            reason = "流動性不足，不建議當沖交易"
        
        return {
            'suitability': suitability,
            'score': liquidity_analysis.score,
            'reason': reason,
            'max_recommended_lots': self.get_optimal_trade_size(stock_data),
            'warnings': liquidity_analysis.warnings
        }