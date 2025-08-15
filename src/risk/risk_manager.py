"""
專業風險管理系統
Professional Risk Management System - 基於專業交易員手冊
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum

from ..models.stock_data import ComprehensiveStockData, MarketContext


class RiskLevel(Enum):
    LOW = "低風險"
    MEDIUM_LOW = "中低風險"
    MEDIUM = "中等風險"
    MEDIUM_HIGH = "中高風險"
    HIGH = "高風險"
    EXTREME = "極高風險"


@dataclass
class RiskAssessment:
    overall_risk_level: RiskLevel
    risk_factors: List[str]
    risk_mitigations: List[str]
    max_position_size: float
    recommended_stop_loss: float
    risk_reward_ratio: str
    special_warnings: List[str]


class ProfessionalRiskManager:
    """專業風險管理器 - 多層次風險評估與控制"""
    
    def __init__(self):
        self.name = "專業風險管理系統"
        
        # 風險控制參數
        self.max_single_trade_risk = 0.02    # 單筆交易最大風險 2%
        self.max_daily_risk = 0.06           # 單日最大風險 6%
        self.max_sector_risk = 0.15          # 單一行業最大風險 15%
        self.max_correlation_risk = 0.25     # 相關性風險上限 25%
        
        # 流動性風險閾值
        self.min_volume_ratio = 1.2          # 最小成交量倍數
        self.min_market_cap = 50             # 最小市值（億元）
        self.max_spread_bps = 30             # 最大買賣價差（基點）
        
        # 市場風險閾值
        self.high_volatility_threshold = 0.06  # 高波動閾值 6%
        self.extreme_rsi_threshold = 15        # 極端RSI閾值
        self.max_gap_risk = 0.08              # 最大跳空風險 8%
        
        # AI模型風險評估
        self.ai_confidence_threshold = 0.75   # AI信心度閾值
        self.min_historical_accuracy = 0.65   # 最小歷史準確率
    
    def assess_comprehensive_risk(self, stock_data: ComprehensiveStockData,
                                 strategy_name: str,
                                 market_context: MarketContext = None,
                                 portfolio_context: Dict = None) -> RiskAssessment:
        """
        全面風險評估
        
        Args:
            stock_data: 完整股票數據
            strategy_name: 策略名稱
            market_context: 市場環境
            portfolio_context: 投資組合環境
            
        Returns:
            RiskAssessment: 風險評估結果
        """
        
        risk_factors = []
        risk_mitigations = []
        special_warnings = []
        
        # 1. 流動性風險評估
        liquidity_risk = self._assess_liquidity_risk(stock_data)
        if liquidity_risk['risk_level'] != 'low':
            risk_factors.extend(liquidity_risk['factors'])
            risk_mitigations.extend(liquidity_risk['mitigations'])
        
        # 2. 市場風險評估
        market_risk = self._assess_market_risk(stock_data, market_context)
        if market_risk['risk_level'] != 'low':
            risk_factors.extend(market_risk['factors'])
            risk_mitigations.extend(market_risk['mitigations'])
        
        # 3. 策略特定風險評估
        strategy_risk = self._assess_strategy_specific_risk(stock_data, strategy_name)
        risk_factors.extend(strategy_risk['factors'])
        risk_mitigations.extend(strategy_risk['mitigations'])
        
        # 4. AI模型風險評估
        ai_risk = self._assess_ai_model_risk(stock_data)
        if ai_risk['risk_level'] != 'low':
            risk_factors.extend(ai_risk['factors'])
            special_warnings.extend(ai_risk['warnings'])
        
        # 5. 時間風險評估
        time_risk = self._assess_time_risk(market_context)
        if time_risk['risk_level'] != 'low':
            risk_factors.extend(time_risk['factors'])
            risk_mitigations.extend(time_risk['mitigations'])
        
        # 6. 投資組合風險評估
        if portfolio_context:
            portfolio_risk = self._assess_portfolio_risk(stock_data, portfolio_context)
            risk_factors.extend(portfolio_risk['factors'])
            risk_mitigations.extend(portfolio_risk['mitigations'])
        
        # 綜合風險等級計算
        overall_risk_level = self._calculate_overall_risk_level(
            liquidity_risk, market_risk, strategy_risk, ai_risk, time_risk
        )
        
        # 計算最大倉位與停損
        max_position_size = self._calculate_max_position_size(overall_risk_level, stock_data)
        recommended_stop_loss = self._calculate_recommended_stop_loss(stock_data, strategy_name)
        risk_reward_ratio = self._calculate_risk_reward_ratio(stock_data, strategy_name)
        
        return RiskAssessment(
            overall_risk_level=overall_risk_level,
            risk_factors=risk_factors,
            risk_mitigations=risk_mitigations,
            max_position_size=max_position_size,
            recommended_stop_loss=recommended_stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            special_warnings=special_warnings
        )
    
    def _assess_liquidity_risk(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """評估流動性風險"""
        
        factors = []
        mitigations = []
        risk_level = 'low'
        
        volume_ratio = stock_data.volume_ratio
        turnover_rate = stock_data.turnover_rate
        
        # 成交量風險
        if volume_ratio < 0.8:
            factors.append("成交量異常萎縮，流動性不足")
            mitigations.append("減少倉位規模，分批進出")
            risk_level = 'high'
        elif volume_ratio < self.min_volume_ratio:
            factors.append("成交量偏低，可能影響執行")
            mitigations.append("降低單筆交易規模")
            risk_level = 'medium'
        
        # 週轉率風險
        if turnover_rate < 0.005:  # 0.5%
            factors.append("週轉率過低，籌碼集中")
            mitigations.append("避免大額交易，使用限價單")
            risk_level = max(risk_level, 'medium')
        
        # 價格跳空風險
        if len(stock_data.price_history) >= 2:
            yesterday_close = stock_data.price_history[-2].close
            today_open = stock_data.price_history[-1].open
            gap_pct = abs(today_open - yesterday_close) / yesterday_close
            
            if gap_pct > self.max_gap_risk:
                factors.append(f"存在{gap_pct:.1%}跳空缺口，流動性風險")
                mitigations.append("等待gap填補或縮小倉位")
                risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'factors': factors,
            'mitigations': mitigations
        }
    
    def _assess_market_risk(self, stock_data: ComprehensiveStockData, 
                           market_context: MarketContext = None) -> Dict[str, Any]:
        """評估市場風險"""
        
        factors = []
        mitigations = []
        risk_level = 'low'
        
        # 波動性風險
        atr_pct = stock_data.atr_percentage
        if atr_pct > self.high_volatility_threshold:
            factors.append(f"波動性極高({atr_pct:.1%})，價格變動劇烈")
            mitigations.append("縮小倉位，設置更嚴格的停損")
            risk_level = 'high'
        elif atr_pct > 0.04:
            factors.append(f"波動性偏高({atr_pct:.1%})")
            mitigations.append("適當降低倉位，密切監控")
            risk_level = 'medium'
        
        # 技術指標極端風險
        rsi = stock_data.technical_indicators.rsi
        if rsi < self.extreme_rsi_threshold or rsi > (100 - self.extreme_rsi_threshold):
            factors.append(f"RSI處於極端區域({rsi:.1f})，反轉風險高")
            mitigations.append("考慮逆向思維，準備快速出場")
            risk_level = max(risk_level, 'medium_high')
        
        # 市場環境風險
        if market_context:
            if market_context.volatility_environment == "high_volatility":
                factors.append("市場整體波動性高")
                mitigations.append("降低整體曝險，增加現金比重")
                risk_level = max(risk_level, 'medium')
            
            if hasattr(market_context, 'vix') and market_context.vix > 35:
                factors.append("恐慌指數極高，市場情緒惡化")
                mitigations.append("考慮暫停交易或僅做防禦性操作")
                risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'factors': factors,
            'mitigations': mitigations
        }
    
    def _assess_strategy_specific_risk(self, stock_data: ComprehensiveStockData, 
                                     strategy_name: str) -> Dict[str, Any]:
        """評估策略特定風險"""
        
        factors = []
        mitigations = []
        
        if strategy_name == 'trend_following':
            # 趨勢策略風險
            if stock_data.ma_trend == 'sideways':
                factors.append("橫盤整理中，趨勢策略風險較高")
                mitigations.append("等待明確突破或轉用其他策略")
            
            momentum_signals = stock_data.get_momentum_signals()
            if len(momentum_signals) < 2:
                factors.append("動能信號不足，趨勢可能轉弱")
                mitigations.append("縮短持有期間，提高警覺")
        
        elif strategy_name == 'counter_trend':
            # 逆勢策略風險
            factors.append("逆勢操作本身具有較高風險")
            mitigations.append("嚴格控制倉位，快進快出")
            
            if stock_data.volume_ratio > 2.0:
                factors.append("成交量暴增，趨勢可能持續")
                mitigations.append("考慮放棄逆勢，順勢而為")
        
        elif strategy_name == 'breakout':
            # 突破策略風險
            if stock_data.volume_ratio < 1.5:
                factors.append("突破缺乏成交量確認")
                mitigations.append("等待量能放大或降低期望")
            
            # 檢查假突破風險
            support_resistance = stock_data.calculate_support_resistance()
            current_price = stock_data.current_price
            resistance = support_resistance['resistance']
            
            if abs(current_price - resistance) / resistance < 0.02:
                factors.append("接近關鍵壓力位，假突破風險")
                mitigations.append("等待明確突破確認")
        
        return {
            'factors': factors,
            'mitigations': mitigations
        }
    
    def _assess_ai_model_risk(self, stock_data: ComprehensiveStockData) -> Dict[str, Any]:
        """評估AI模型風險"""
        
        factors = []
        warnings = []
        risk_level = 'low'
        
        # AI預測不確定性
        # 註：實際應用中需要從AI模型獲取信心度分數
        factors.append("AI模型存在固有的不確定性")
        warnings.append("AI建議僅供參考，不應完全依賴")
        
        # 數據品質風險
        if stock_data.volume_ratio < 0.5:
            factors.append("數據異常，AI分析可能不準確")
            warnings.append("數據品質異常，建議人工複核")
            risk_level = 'medium'
        
        # 模型適用性風險
        atr_pct = stock_data.atr_percentage
        if atr_pct > 0.08:  # 超高波動
            factors.append("極端市況下，AI模型準確性可能下降")
            warnings.append("極端波動環境，增加人工判斷比重")
            risk_level = 'medium'
        
        return {
            'risk_level': risk_level,
            'factors': factors,
            'warnings': warnings
        }
    
    def _assess_time_risk(self, market_context: MarketContext = None) -> Dict[str, Any]:
        """評估時間風險"""
        
        factors = []
        mitigations = []
        risk_level = 'low'
        
        if market_context and market_context.trading_session:
            session = market_context.trading_session
            
            if session == "opening":
                factors.append("開盤時段波動較大")
                mitigations.append("密切關注開盤價格行為")
                risk_level = 'medium'
            
            elif session == "closing":
                factors.append("收盤前強制平倉風險")
                mitigations.append("提前規劃出場，避免被動平倉")
                risk_level = 'medium_high'
            
            elif session == "lunch_break":
                factors.append("午休時段流動性不足")
                mitigations.append("避免在此時段進行大額交易")
                risk_level = 'medium'
        
        return {
            'risk_level': risk_level,
            'factors': factors,
            'mitigations': mitigations
        }
    
    def _assess_portfolio_risk(self, stock_data: ComprehensiveStockData, 
                              portfolio_context: Dict) -> Dict[str, Any]:
        """評估投資組合風險"""
        
        factors = []
        mitigations = []
        
        # 集中度風險
        if portfolio_context.get('position_count', 0) < 3:
            factors.append("投資組合過度集中")
            mitigations.append("考慮分散投資至更多標的")
        
        # 相關性風險
        sector_exposure = portfolio_context.get('sector_exposure', {})
        max_sector_pct = max(sector_exposure.values()) if sector_exposure else 0
        
        if max_sector_pct > self.max_sector_risk:
            factors.append(f"行業集中度過高({max_sector_pct:.1%})")
            mitigations.append("減少同行業曝險")
        
        # 整體風險曝險
        total_risk_exposure = portfolio_context.get('total_risk_exposure', 0)
        if total_risk_exposure > self.max_daily_risk:
            factors.append("整體風險曝險過高")
            mitigations.append("暫停新增倉位")
        
        return {
            'factors': factors,
            'mitigations': mitigations
        }
    
    def _calculate_overall_risk_level(self, *risk_assessments) -> RiskLevel:
        """計算整體風險等級"""
        
        risk_score = 0
        
        for assessment in risk_assessments:
            risk_level = assessment.get('risk_level', 'low')
            
            if risk_level == 'low':
                risk_score += 1
            elif risk_level == 'medium':
                risk_score += 2
            elif risk_level == 'medium_high':
                risk_score += 3
            elif risk_level == 'high':
                risk_score += 4
        
        # 計算平均風險分數
        avg_risk = risk_score / len(risk_assessments)
        
        if avg_risk <= 1.2:
            return RiskLevel.LOW
        elif avg_risk <= 1.8:
            return RiskLevel.MEDIUM_LOW
        elif avg_risk <= 2.5:
            return RiskLevel.MEDIUM
        elif avg_risk <= 3.2:
            return RiskLevel.MEDIUM_HIGH
        else:
            return RiskLevel.HIGH
    
    def _calculate_max_position_size(self, risk_level: RiskLevel, 
                                   stock_data: ComprehensiveStockData) -> float:
        """計算最大倉位規模"""
        
        base_position_pct = 0.1  # 基礎倉位 10%
        
        # 根據風險等級調整
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM_LOW: 0.8,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.MEDIUM_HIGH: 0.4,
            RiskLevel.HIGH: 0.2,
            RiskLevel.EXTREME: 0.1
        }
        
        adjusted_position = base_position_pct * risk_multipliers[risk_level]
        
        # 根據波動性進一步調整
        atr_pct = stock_data.atr_percentage
        volatility_adjustment = min(1.0, 0.03 / atr_pct) if atr_pct > 0 else 1.0
        
        final_position = adjusted_position * volatility_adjustment
        
        return min(0.15, max(0.02, final_position))  # 限制在2%-15%之間
    
    def _calculate_recommended_stop_loss(self, stock_data: ComprehensiveStockData, 
                                       strategy_name: str) -> float:
        """計算建議停損點"""
        
        current_price = stock_data.current_price
        atr = stock_data.technical_indicators.atr
        
        # 基於ATR的動態停損
        if strategy_name == 'trend_following':
            stop_loss_distance = atr * 1.5  # 趨勢策略給予較大空間
        elif strategy_name == 'counter_trend':
            stop_loss_distance = atr * 1.0  # 逆勢策略較嚴格
        else:  # breakout
            stop_loss_distance = atr * 1.2  # 突破策略中等
        
        # 技術支撐調整
        support_resistance = stock_data.calculate_support_resistance()
        support = support_resistance['support']
        
        # 選擇較保守的停損點
        technical_stop = support * 0.99
        atr_stop = current_price - stop_loss_distance
        
        return max(technical_stop, atr_stop)
    
    def _calculate_risk_reward_ratio(self, stock_data: ComprehensiveStockData, 
                                   strategy_name: str) -> str:
        """計算風險回報比"""
        
        current_price = stock_data.current_price
        atr = stock_data.technical_indicators.atr
        
        # 估算潛在利潤
        if strategy_name == 'trend_following':
            potential_profit = atr * 2.5
        elif strategy_name == 'counter_trend':
            potential_profit = atr * 1.8
        else:  # breakout
            potential_profit = atr * 2.2
        
        # 估算風險
        potential_risk = atr * 1.2
        
        ratio = potential_profit / potential_risk if potential_risk > 0 else 0
        
        if ratio >= 2.5:
            return "優秀(>2.5:1)"
        elif ratio >= 2.0:
            return "良好(2:1-2.5:1)"
        elif ratio >= 1.5:
            return "可接受(1.5:1-2:1)"
        else:
            return "偏低(<1.5:1)"
    
    def generate_risk_report(self, risk_assessment: RiskAssessment, 
                           stock_symbol: str) -> Dict[str, Any]:
        """生成風險評估報告"""
        
        return {
            'stock_symbol': stock_symbol,
            'risk_summary': {
                'overall_risk_level': risk_assessment.overall_risk_level.value,
                'max_position_size': f"{risk_assessment.max_position_size:.1%}",
                'recommended_stop_loss': f"{risk_assessment.recommended_stop_loss:.2f}",
                'risk_reward_ratio': risk_assessment.risk_reward_ratio
            },
            'risk_factors': risk_assessment.risk_factors,
            'risk_mitigations': risk_assessment.risk_mitigations,
            'special_warnings': risk_assessment.special_warnings,
            'recommendations': [
                f"建議最大倉位: {risk_assessment.max_position_size:.1%}",
                f"建議停損點: {risk_assessment.recommended_stop_loss:.2f}",
                "嚴格遵守風險管理紀律",
                "密切監控市場變化"
            ]
        }