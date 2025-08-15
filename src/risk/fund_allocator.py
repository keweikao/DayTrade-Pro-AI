"""
資金配置計算器
Fund Allocation Calculator - 專業倉位管理與資金分配
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum

from ..models.stock_data import ComprehensiveStockData, MarketContext
from .risk_manager import RiskAssessment, RiskLevel


class PositionSizeMethod(Enum):
    FIXED_PERCENTAGE = "固定百分比"
    ATR_BASED = "ATR風險調整"
    VOLATILITY_TARGET = "波動率目標"
    KELLY_CRITERION = "凱利公式"
    EQUAL_RISK = "等風險分配"


@dataclass
class PositionCalculation:
    recommended_shares: int
    position_value: float
    position_percentage: float
    risk_amount: float
    stop_loss_price: float
    position_size_method: PositionSizeMethod
    calculation_details: Dict[str, Any]


class FundAllocationCalculator:
    """專業資金配置計算器"""
    
    def __init__(self, total_capital: float = 1000000):
        self.total_capital = total_capital
        self.available_capital = total_capital
        self.reserved_cash_ratio = 0.1  # 保留10%現金
        
        # 風險參數
        self.max_single_position = 0.15    # 單一持股最大15%
        self.max_risk_per_trade = 0.02     # 單筆最大風險2%
        self.max_daily_risk = 0.06         # 日風險上限6%
        self.target_volatility = 0.15      # 目標組合波動率15%
        
        # Kelly公式參數
        self.kelly_multiplier = 0.25       # Kelly公式保守係數
        self.min_win_rate = 0.55           # 最小勝率要求
        
        # 當沖特殊參數
        self.intraday_leverage = 2.5       # 當沖槓桿倍數
        self.intraday_risk_multiplier = 1.5 # 當沖風險係數
    
    def calculate_position_size(self, 
                              stock_data: ComprehensiveStockData,
                              risk_assessment: RiskAssessment,
                              strategy_info: Dict[str, Any],
                              method: PositionSizeMethod = PositionSizeMethod.ATR_BASED,
                              custom_params: Dict = None) -> PositionCalculation:
        """
        計算建議倉位大小
        
        Args:
            stock_data: 股票數據
            risk_assessment: 風險評估
            strategy_info: 策略資訊
            method: 計算方法
            custom_params: 自定義參數
            
        Returns:
            PositionCalculation: 倉位計算結果
        """
        
        current_price = stock_data.current_price
        stop_loss_price = risk_assessment.recommended_stop_loss
        
        if method == PositionSizeMethod.FIXED_PERCENTAGE:
            result = self._calculate_fixed_percentage(
                current_price, risk_assessment.max_position_size
            )
        elif method == PositionSizeMethod.ATR_BASED:
            result = self._calculate_atr_based(
                stock_data, stop_loss_price, risk_assessment
            )
        elif method == PositionSizeMethod.VOLATILITY_TARGET:
            result = self._calculate_volatility_target(
                stock_data, risk_assessment
            )
        elif method == PositionSizeMethod.KELLY_CRITERION:
            result = self._calculate_kelly_criterion(
                stock_data, strategy_info, risk_assessment
            )
        else:  # EQUAL_RISK
            result = self._calculate_equal_risk(
                stock_data, stop_loss_price, risk_assessment
            )
        
        # 應用資金管理約束
        result = self._apply_capital_constraints(result, current_price)
        
        # 當沖調整
        if strategy_info.get('is_intraday', True):
            result = self._apply_intraday_adjustments(result, current_price)
        
        return result
    
    def _calculate_fixed_percentage(self, current_price: float, 
                                  max_position_pct: float) -> PositionCalculation:
        """固定百分比法"""
        
        available_funds = self.available_capital * (1 - self.reserved_cash_ratio)
        position_value = available_funds * max_position_pct
        recommended_shares = int(position_value / current_price)
        actual_position_value = recommended_shares * current_price
        actual_position_pct = actual_position_value / self.total_capital
        
        return PositionCalculation(
            recommended_shares=recommended_shares,
            position_value=actual_position_value,
            position_percentage=actual_position_pct,
            risk_amount=actual_position_value * 0.02,  # 假設2%風險
            stop_loss_price=current_price * 0.98,      # 假設2%停損
            position_size_method=PositionSizeMethod.FIXED_PERCENTAGE,
            calculation_details={
                'method': '固定百分比',
                'target_percentage': max_position_pct,
                'available_funds': available_funds
            }
        )
    
    def _calculate_atr_based(self, stock_data: ComprehensiveStockData,
                           stop_loss_price: float,
                           risk_assessment: RiskAssessment) -> PositionCalculation:
        """ATR風險調整法"""
        
        current_price = stock_data.current_price
        atr = stock_data.technical_indicators.atr
        
        # 計算每股風險
        risk_per_share = current_price - stop_loss_price
        
        # 基於總資金的風險金額
        max_risk_amount = self.total_capital * self.max_risk_per_trade
        
        # 計算股數
        if risk_per_share > 0:
            recommended_shares = int(max_risk_amount / risk_per_share)
        else:
            recommended_shares = 0
        
        position_value = recommended_shares * current_price
        position_percentage = position_value / self.total_capital
        
        # ATR調整係數
        atr_adjustment = min(1.0, 0.02 / stock_data.atr_percentage) if stock_data.atr_percentage > 0 else 1.0
        recommended_shares = int(recommended_shares * atr_adjustment)
        
        return PositionCalculation(
            recommended_shares=recommended_shares,
            position_value=recommended_shares * current_price,
            position_percentage=position_percentage,
            risk_amount=recommended_shares * risk_per_share,
            stop_loss_price=stop_loss_price,
            position_size_method=PositionSizeMethod.ATR_BASED,
            calculation_details={
                'method': 'ATR風險調整',
                'atr': atr,
                'atr_percentage': stock_data.atr_percentage,
                'risk_per_share': risk_per_share,
                'max_risk_amount': max_risk_amount,
                'atr_adjustment': atr_adjustment
            }
        )
    
    def _calculate_volatility_target(self, stock_data: ComprehensiveStockData,
                                   risk_assessment: RiskAssessment) -> PositionCalculation:
        """波動率目標法"""
        
        current_price = stock_data.current_price
        stock_volatility = stock_data.atr_percentage
        
        # 目標倉位 = 目標波動率 / 股票波動率
        if stock_volatility > 0:
            target_position_pct = min(
                self.target_volatility / stock_volatility,
                risk_assessment.max_position_size
            )
        else:
            target_position_pct = risk_assessment.max_position_size
        
        available_funds = self.available_capital * (1 - self.reserved_cash_ratio)
        position_value = available_funds * target_position_pct
        recommended_shares = int(position_value / current_price)
        
        return PositionCalculation(
            recommended_shares=recommended_shares,
            position_value=recommended_shares * current_price,
            position_percentage=target_position_pct,
            risk_amount=recommended_shares * current_price * stock_volatility,
            stop_loss_price=risk_assessment.recommended_stop_loss,
            position_size_method=PositionSizeMethod.VOLATILITY_TARGET,
            calculation_details={
                'method': '波動率目標',
                'target_volatility': self.target_volatility,
                'stock_volatility': stock_volatility,
                'volatility_ratio': self.target_volatility / stock_volatility if stock_volatility > 0 else 0
            }
        )
    
    def _calculate_kelly_criterion(self, stock_data: ComprehensiveStockData,
                                 strategy_info: Dict[str, Any],
                                 risk_assessment: RiskAssessment) -> PositionCalculation:
        """Kelly公式法"""
        
        current_price = stock_data.current_price
        
        # 從策略資訊獲取勝率和賠率
        win_rate = strategy_info.get('expected_win_rate', 0.6)
        if isinstance(win_rate, str):
            # 轉換文字描述為數值
            win_rate_mapping = {
                '極高': 0.75, '高': 0.7, '中高': 0.65,
                '中等': 0.6, '中等偏低': 0.55, '偏低': 0.5, '低': 0.45
            }
            win_rate = win_rate_mapping.get(win_rate, 0.6)
        
        # 估算平均賠率（基於ATR）
        atr = stock_data.technical_indicators.atr
        avg_win = atr * 2.0  # 假設平均獲利2倍ATR
        avg_loss = atr * 1.2  # 假設平均虧損1.2倍ATR
        
        if avg_loss > 0:
            odds_ratio = avg_win / avg_loss
        else:
            odds_ratio = 2.0
        
        # Kelly公式: f = (bp - q) / b
        # b = odds_ratio, p = win_rate, q = 1 - win_rate
        if win_rate >= self.min_win_rate and odds_ratio > 1:
            kelly_fraction = (odds_ratio * win_rate - (1 - win_rate)) / odds_ratio
            kelly_fraction = max(0, min(kelly_fraction * self.kelly_multiplier, 0.25))
        else:
            kelly_fraction = 0.05  # 最小倉位
        
        position_value = self.available_capital * kelly_fraction
        recommended_shares = int(position_value / current_price)
        
        return PositionCalculation(
            recommended_shares=recommended_shares,
            position_value=recommended_shares * current_price,
            position_percentage=kelly_fraction,
            risk_amount=recommended_shares * avg_loss,
            stop_loss_price=risk_assessment.recommended_stop_loss,
            position_size_method=PositionSizeMethod.KELLY_CRITERION,
            calculation_details={
                'method': 'Kelly公式',
                'win_rate': win_rate,
                'odds_ratio': odds_ratio,
                'kelly_fraction': kelly_fraction,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
        )
    
    def _calculate_equal_risk(self, stock_data: ComprehensiveStockData,
                            stop_loss_price: float,
                            risk_assessment: RiskAssessment) -> PositionCalculation:
        """等風險分配法"""
        
        current_price = stock_data.current_price
        risk_per_share = current_price - stop_loss_price
        
        # 每個倉位承擔相同的風險金額
        risk_per_position = self.total_capital * (self.max_risk_per_trade * 0.8)  # 略為保守
        
        if risk_per_share > 0:
            recommended_shares = int(risk_per_position / risk_per_share)
        else:
            recommended_shares = 0
        
        return PositionCalculation(
            recommended_shares=recommended_shares,
            position_value=recommended_shares * current_price,
            position_percentage=recommended_shares * current_price / self.total_capital,
            risk_amount=risk_per_position,
            stop_loss_price=stop_loss_price,
            position_size_method=PositionSizeMethod.EQUAL_RISK,
            calculation_details={
                'method': '等風險分配',
                'risk_per_position': risk_per_position,
                'risk_per_share': risk_per_share
            }
        )
    
    def _apply_capital_constraints(self, calculation: PositionCalculation,
                                 current_price: float) -> PositionCalculation:
        """應用資金管理約束"""
        
        # 最大單一持股限制
        max_position_value = self.total_capital * self.max_single_position
        if calculation.position_value > max_position_value:
            calculation.recommended_shares = int(max_position_value / current_price)
            calculation.position_value = calculation.recommended_shares * current_price
            calculation.position_percentage = calculation.position_value / self.total_capital
        
        # 可用資金限制
        available_funds = self.available_capital * (1 - self.reserved_cash_ratio)
        if calculation.position_value > available_funds:
            calculation.recommended_shares = int(available_funds / current_price)
            calculation.position_value = calculation.recommended_shares * current_price
            calculation.position_percentage = calculation.position_value / self.total_capital
        
        # 最小交易單位（1000股）
        calculation.recommended_shares = max(1000, 
            (calculation.recommended_shares // 1000) * 1000)
        calculation.position_value = calculation.recommended_shares * current_price
        calculation.position_percentage = calculation.position_value / self.total_capital
        
        return calculation
    
    def _apply_intraday_adjustments(self, calculation: PositionCalculation,
                                  current_price: float) -> PositionCalculation:
        """當沖調整"""
        
        # 當沖可以使用槓桿
        leveraged_shares = int(calculation.recommended_shares * self.intraday_leverage)
        
        # 但要考慮風險係數
        risk_adjusted_shares = int(leveraged_shares / self.intraday_risk_multiplier)
        
        # 選擇較保守的數量
        final_shares = min(leveraged_shares, risk_adjusted_shares)
        
        calculation.recommended_shares = final_shares
        calculation.position_value = final_shares * current_price
        calculation.position_percentage = calculation.position_value / self.total_capital
        
        # 更新計算詳情
        calculation.calculation_details.update({
            'intraday_leverage': self.intraday_leverage,
            'risk_multiplier': self.intraday_risk_multiplier,
            'leveraged_shares': leveraged_shares,
            'risk_adjusted_shares': risk_adjusted_shares
        })
        
        return calculation
    
    def calculate_portfolio_allocation(self, 
                                     candidates: List[Dict[str, Any]],
                                     max_positions: int = 5) -> Dict[str, Any]:
        """
        計算投資組合配置
        
        Args:
            candidates: 候選股票列表
            max_positions: 最大持股數量
            
        Returns:
            Dict: 投資組合配置建議
        """
        
        # 按分數排序候選股票
        sorted_candidates = sorted(candidates, 
                                 key=lambda x: x.get('score', 0), reverse=True)
        
        # 選擇前N個標的
        selected_stocks = sorted_candidates[:max_positions]
        
        # 計算每個標的的配置
        allocations = []
        total_allocated = 0
        
        for stock in selected_stocks:
            stock_data = stock['stock_data']
            risk_assessment = stock['risk_assessment']
            strategy_info = stock['strategy_info']
            
            # 計算倉位
            position_calc = self.calculate_position_size(
                stock_data, risk_assessment, strategy_info,
                method=PositionSizeMethod.EQUAL_RISK
            )
            
            allocations.append({
                'symbol': stock.get('symbol', ''),
                'position_calculation': position_calc,
                'strategy': strategy_info.get('strategy_name', ''),
                'score': stock.get('score', 0)
            })
            
            total_allocated += position_calc.position_value
        
        # 計算組合風險指標
        portfolio_risk = self._calculate_portfolio_risk(allocations)
        
        return {
            'allocations': allocations,
            'portfolio_summary': {
                'total_allocated': total_allocated,
                'cash_reserved': self.total_capital - total_allocated,
                'utilization_rate': total_allocated / self.total_capital,
                'number_of_positions': len(allocations)
            },
            'risk_metrics': portfolio_risk,
            'recommendations': self._generate_allocation_recommendations(allocations, portfolio_risk)
        }
    
    def _calculate_portfolio_risk(self, allocations: List[Dict]) -> Dict[str, Any]:
        """計算投資組合風險"""
        
        total_risk = sum(alloc['position_calculation'].risk_amount for alloc in allocations)
        total_value = sum(alloc['position_calculation'].position_value for alloc in allocations)
        
        # 集中度風險
        if len(allocations) > 0:
            max_position_pct = max(alloc['position_calculation'].position_percentage 
                                 for alloc in allocations)
            avg_position_pct = total_value / self.total_capital / len(allocations)
        else:
            max_position_pct = 0
            avg_position_pct = 0
        
        return {
            'total_portfolio_risk': total_risk,
            'risk_percentage': total_risk / self.total_capital,
            'max_position_concentration': max_position_pct,
            'average_position_size': avg_position_pct,
            'diversification_score': 1 - (max_position_pct / avg_position_pct) if avg_position_pct > 0 else 0
        }
    
    def _generate_allocation_recommendations(self, allocations: List[Dict],
                                           portfolio_risk: Dict) -> List[str]:
        """生成配置建議"""
        
        recommendations = []
        
        # 風險檢查
        if portfolio_risk['risk_percentage'] > self.max_daily_risk:
            recommendations.append("投資組合整體風險過高，建議降低倉位")
        
        # 集中度檢查
        if portfolio_risk['max_position_concentration'] > self.max_single_position:
            recommendations.append("單一持股比重過高，建議分散投資")
        
        # 多樣化檢查
        if portfolio_risk['diversification_score'] < 0.5:
            recommendations.append("投資組合集中度過高，缺乏分散效果")
        
        # 現金比例檢查
        cash_ratio = 1 - (portfolio_risk.get('total_portfolio_risk', 0) / self.total_capital)
        if cash_ratio < self.reserved_cash_ratio:
            recommendations.append(f"建議保留至少{self.reserved_cash_ratio:.0%}現金")
        
        if not recommendations:
            recommendations.append("投資組合配置合理，風險控制良好")
        
        return recommendations
    
    def update_available_capital(self, new_capital: float):
        """更新可用資金"""
        self.available_capital = new_capital
    
    def get_position_sizing_summary(self) -> Dict[str, Any]:
        """獲取倉位管理摘要"""
        
        return {
            'capital_management': {
                'total_capital': self.total_capital,
                'available_capital': self.available_capital,
                'reserved_cash_ratio': self.reserved_cash_ratio,
                'max_single_position': self.max_single_position
            },
            'risk_parameters': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_daily_risk': self.max_daily_risk,
                'target_volatility': self.target_volatility
            },
            'intraday_settings': {
                'leverage': self.intraday_leverage,
                'risk_multiplier': self.intraday_risk_multiplier
            },
            'methods_available': [method.value for method in PositionSizeMethod]
        }