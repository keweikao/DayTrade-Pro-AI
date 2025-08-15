"""
策略識別引擎
Strategy Identification Engine - 基於專業交易員手冊的三大策略
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..models.stock_data import (
    ComprehensiveStockData, MarketContext, StrategyRecommendation
)
from .trend_following import TrendFollowingStrategy
from .counter_trend import CounterTrendStrategy  
from .breakout_strategy import BreakoutStrategy


class StrategyIdentifier:
    """策略識別器 - 自動選擇最適合的交易策略"""
    
    def __init__(self):
        self.strategies = {
            'trend_following': TrendFollowingStrategy(),
            'counter_trend': CounterTrendStrategy(),
            'breakout': BreakoutStrategy()
        }
        
        # 策略適用的市場條件權重
        self.market_condition_weights = {
            'trend_following': {
                'trending_market': 1.0,
                'high_volume': 0.8,
                'clear_direction': 0.9,
                'momentum_signals': 0.8
            },
            'counter_trend': {
                'ranging_market': 1.0,
                'oversold_overbought': 0.9,
                'high_volatility': 0.7,
                'divergence_signals': 0.8
            },
            'breakout': {
                'consolidation_pattern': 1.0,
                'volume_surge': 0.9,
                'key_level_test': 0.8,
                'low_to_high_volatility': 0.7
            }
        }
    
    def identify_optimal_strategy(self, stock_data: ComprehensiveStockData, 
                                 market_context: MarketContext = None) -> Dict[str, Any]:
        """
        識別最佳交易策略
        
        Args:
            stock_data: 完整股票數據
            market_context: 市場環境（可選）
            
        Returns:
            Dict: 策略推薦結果
        """
        
        strategy_scores = {}
        strategy_evaluations = {}
        
        # 評估每個策略的適用性
        for strategy_name, strategy in self.strategies.items():
            evaluation = strategy.evaluate_suitability(stock_data, market_context)
            strategy_scores[strategy_name] = evaluation['score']
            strategy_evaluations[strategy_name] = evaluation
        
        # 找出最佳策略
        best_strategy_name = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy_name]
        
        # 檢查分數是否達到最低閾值
        if best_score < 50:
            return {
                'recommended_strategy': None,
                'reason': '當前市況無明確策略適用，建議觀望',
                'all_scores': strategy_scores,
                'market_analysis': self._analyze_market_conditions(stock_data, market_context)
            }
        
        # 生成策略執行計劃
        best_strategy = self.strategies[best_strategy_name]
        execution_plan = best_strategy.generate_execution_plan(stock_data, market_context)
        
        # 檢查策略衝突
        strategy_conflicts = self._check_strategy_conflicts(strategy_scores, strategy_evaluations)
        
        return {
            'recommended_strategy': best_strategy_name,
            'strategy_score': best_score,
            'execution_plan': execution_plan,
            'strategy_evaluation': strategy_evaluations[best_strategy_name],
            'all_scores': strategy_scores,
            'alternative_strategies': self._get_alternative_strategies(strategy_scores, best_strategy_name),
            'strategy_conflicts': strategy_conflicts,
            'confidence_level': self._calculate_confidence_level(best_score, strategy_scores),
            'market_suitability': self._assess_market_suitability(stock_data, market_context, best_strategy_name)
        }
    
    def _analyze_market_conditions(self, stock_data: ComprehensiveStockData, 
                                  market_context: MarketContext = None) -> Dict[str, Any]:
        """分析市場條件"""
        conditions = {}
        
        # 趨勢分析
        if stock_data.ma_trend == "bullish":
            conditions['trend'] = "上升趨勢"
        elif stock_data.ma_trend == "bearish":
            conditions['trend'] = "下降趨勢"
        else:
            conditions['trend'] = "盤整趨勢"
        
        # 波動性分析
        atr_pct = stock_data.atr_percentage
        if atr_pct > 0.04:
            conditions['volatility'] = "高波動"
        elif atr_pct > 0.02:
            conditions['volatility'] = "中波動"
        else:
            conditions['volatility'] = "低波動"
        
        # 成交量分析
        volume_ratio = stock_data.volume_ratio
        if volume_ratio > 2.0:
            conditions['volume'] = "量能爆發"
        elif volume_ratio > 1.5:
            conditions['volume'] = "量能放大"
        else:
            conditions['volume'] = "量能正常"
        
        # 技術指標狀態
        rsi = stock_data.technical_indicators.rsi
        if rsi > 70:
            conditions['momentum'] = "超買"
        elif rsi < 30:
            conditions['momentum'] = "超賣"
        else:
            conditions['momentum'] = "中性"
        
        return conditions
    
    def _check_strategy_conflicts(self, strategy_scores: Dict[str, float], 
                                 strategy_evaluations: Dict[str, Dict]) -> List[str]:
        """檢查策略衝突"""
        conflicts = []
        
        # 如果多個策略分數接近，可能存在衝突
        sorted_scores = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_scores) >= 2:
            best_score = sorted_scores[0][1]
            second_score = sorted_scores[1][1]
            
            # 如果分數差距小於15分，視為衝突
            if best_score - second_score < 15:
                conflicts.append(f"多重策略信號: {sorted_scores[0][0]}({best_score:.0f}) vs {sorted_scores[1][0]}({second_score:.0f})")
        
        # 檢查順勢與逆勢策略衝突
        trend_score = strategy_scores.get('trend_following', 0)
        counter_score = strategy_scores.get('counter_trend', 0)
        
        if trend_score > 60 and counter_score > 60:
            conflicts.append("順勢與逆勢策略同時適用，市場可能處於轉折點")
        
        return conflicts
    
    def _get_alternative_strategies(self, strategy_scores: Dict[str, float], 
                                   best_strategy: str) -> List[Dict[str, Any]]:
        """獲取備選策略"""
        alternatives = []
        
        for strategy_name, score in strategy_scores.items():
            if strategy_name != best_strategy and score >= 50:
                alternatives.append({
                    'strategy': strategy_name,
                    'score': score,
                    'reason': f"次選方案，適用度{score:.0f}分"
                })
        
        # 按分數排序
        alternatives.sort(key=lambda x: x['score'], reverse=True)
        
        return alternatives[:2]  # 最多返回2個備選策略
    
    def _calculate_confidence_level(self, best_score: float, all_scores: Dict[str, float]) -> str:
        """計算信心等級"""
        
        # 計算分數差距
        sorted_scores = sorted(all_scores.values(), reverse=True)
        score_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else best_score
        
        if best_score >= 85 and score_gap >= 20:
            return "極高"
        elif best_score >= 75 and score_gap >= 15:
            return "高"
        elif best_score >= 65 and score_gap >= 10:
            return "中"
        elif best_score >= 55:
            return "偏低"
        else:
            return "低"
    
    def _assess_market_suitability(self, stock_data: ComprehensiveStockData, 
                                  market_context: MarketContext,
                                  strategy_name: str) -> Dict[str, Any]:
        """評估市場適用性"""
        
        suitability_factors = []
        overall_score = 70  # 基礎分數
        
        # 大盤環境適用性
        if market_context:
            if strategy_name == 'trend_following':
                if market_context.is_market_bullish:
                    suitability_factors.append("大盤配合多頭策略")
                    overall_score += 10
                else:
                    suitability_factors.append("大盤偏空，順勢做空機會")
                    overall_score += 5
            
            elif strategy_name == 'counter_trend':
                if market_context.volatility_environment == "high_volatility":
                    suitability_factors.append("高波動環境適合逆勢操作")
                    overall_score += 10
                else:
                    overall_score -= 5
            
            elif strategy_name == 'breakout':
                if market_context.volatility_environment in ["low_volatility", "medium_volatility"]:
                    suitability_factors.append("適合突破策略的波動環境")
                    overall_score += 10
        
        # 個股特性適用性
        liquidity_suitable = stock_data.volume_ratio > 1.3
        volatility_suitable = stock_data.atr_percentage > 0.02
        
        if liquidity_suitable:
            suitability_factors.append("流動性充足")
            overall_score += 5
        
        if volatility_suitable:
            suitability_factors.append("波動性適中")
            overall_score += 5
        
        # 時段適用性
        if market_context and market_context.trading_session:
            session = market_context.trading_session
            
            if session == "opening" and strategy_name == "breakout":
                suitability_factors.append("開盤時段適合突破策略")
                overall_score += 5
            elif session == "trading" and strategy_name == "trend_following":
                suitability_factors.append("盤中時段適合順勢策略")
                overall_score += 5
            elif session == "closing":
                suitability_factors.append("收盤前風險較高")
                overall_score -= 10
        
        return {
            'overall_score': min(100, max(0, overall_score)),
            'suitability_factors': suitability_factors,
            'market_timing': "適合" if overall_score >= 70 else "謹慎" if overall_score >= 60 else "不適合"
        }
    
    def get_strategy_comparison(self, stock_data: ComprehensiveStockData, 
                               market_context: MarketContext = None) -> Dict[str, Any]:
        """獲取策略比較分析"""
        
        comparison_result = {}
        
        for strategy_name, strategy in self.strategies.items():
            evaluation = strategy.evaluate_suitability(stock_data, market_context)
            
            comparison_result[strategy_name] = {
                'score': evaluation['score'],
                'pros': evaluation.get('advantages', []),
                'cons': evaluation.get('disadvantages', []),
                'risk_level': evaluation.get('risk_level', '中'),
                'win_rate_expectation': evaluation.get('expected_win_rate', '中等'),
                'profit_potential': evaluation.get('expected_risk_reward', '中等'),
                'suitable_conditions': evaluation.get('conditions_met', [])
            }
        
        # 添加總體分析
        comparison_result['overall_analysis'] = {
            'market_conditions': self._analyze_market_conditions(stock_data, market_context),
            'recommended_approach': self._suggest_overall_approach(comparison_result),
            'risk_considerations': self._identify_risk_considerations(stock_data, market_context)
        }
        
        return comparison_result
    
    def _suggest_overall_approach(self, comparison_result: Dict[str, Any]) -> str:
        """建議整體操作方針"""
        
        scores = {name: data['score'] for name, data in comparison_result.items() 
                 if name != 'overall_analysis'}
        
        max_score = max(scores.values())
        high_score_strategies = [name for name, score in scores.items() if score >= max_score - 10]
        
        if max_score < 50:
            return "當前市況不明朗，建議觀望等待更清晰的信號"
        elif len(high_score_strategies) == 1:
            return f"市況明確適合{high_score_strategies[0]}策略，建議專注執行"
        else:
            return f"多重策略適用，可考慮組合操作或等待更明確的信號"
    
    def _identify_risk_considerations(self, stock_data: ComprehensiveStockData, 
                                    market_context: MarketContext = None) -> List[str]:
        """識別風險考量"""
        
        risks = []
        
        # 流動性風險
        if stock_data.volume_ratio < 1.2:
            risks.append("成交量不足，流動性風險較高")
        
        # 波動性風險
        if stock_data.atr_percentage > 0.06:
            risks.append("波動性極高，價格變動劇烈")
        elif stock_data.atr_percentage < 0.015:
            risks.append("波動性過低，獲利空間有限")
        
        # 技術指標風險
        rsi = stock_data.technical_indicators.rsi
        if rsi > 80:
            risks.append("RSI極度超買，回調風險高")
        elif rsi < 20:
            risks.append("RSI極度超賣，可能持續弱勢")
        
        # 市場環境風險
        if market_context:
            if market_context.volatility_environment == "high_volatility":
                risks.append("市場高波動環境，系統性風險增加")
            
            if market_context.trading_session == "closing":
                risks.append("接近收盤，強制平倉風險")
        
        return risks
    
    def validate_strategy_execution(self, strategy_name: str, stock_data: ComprehensiveStockData,
                                   execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """驗證策略執行可行性"""
        
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'adjustments': [],
            'execution_feasibility': 'high'
        }
        
        # 檢查執行條件
        if 'entry_scenarios' in execution_plan:
            for scenario in execution_plan['entry_scenarios']:
                entry_price = scenario.get('entry_price', 0)
                current_price = stock_data.current_price
                
                # 檢查進場價格合理性
                price_diff_pct = abs(entry_price - current_price) / current_price
                if price_diff_pct > 0.05:  # 5%差距
                    validation_result['warnings'].append(
                        f"進場價格{entry_price:.2f}與現價{current_price:.2f}差距較大({price_diff_pct:.1%})"
                    )
        
        # 檢查止損設置
        if 'stop_loss' in execution_plan:
            stop_loss = execution_plan['stop_loss']
            current_price = stock_data.current_price
            
            risk_pct = abs(current_price - stop_loss) / current_price
            if risk_pct > 0.05:  # 風險超過5%
                validation_result['warnings'].append(f"單筆風險過高: {risk_pct:.1%}")
                validation_result['adjustments'].append("建議縮小止損距離或減少倉位")
        
        # 檢查成交量支撐
        if stock_data.volume_ratio < 1.3:
            validation_result['warnings'].append("成交量不足，執行可能面臨滑價")
            validation_result['execution_feasibility'] = 'medium'
        
        # 根據警告數量調整可行性
        if len(validation_result['warnings']) >= 3:
            validation_result['execution_feasibility'] = 'low'
            validation_result['is_valid'] = False
        
        return validation_result