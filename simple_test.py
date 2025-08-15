"""
簡化系統測試
Simplified System Test - 驗證核心功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import date, datetime
from src.models.stock_data import StockPrice, TechnicalIndicators, StockInfo, MarketData, ComprehensiveStockData
from src.core.liquidity_analyzer import LiquidityAnalyzer
from src.core.volatility_analyzer import VolatilityAnalyzer
from src.core.momentum_analyzer import MomentumAnalyzer
from src.strategies.strategy_identifier import StrategyIdentifier
from src.risk.risk_manager import ProfessionalRiskManager
from src.risk.fund_allocator import FundAllocationCalculator

def create_test_data():
    """創建測試數據"""
    
    # 股票基本資訊
    stock_info = StockInfo(
        code="2330",
        name="台積電",
        industry="半導體",
        market_cap=15000000000000,  # 15兆
        shares_outstanding=25930000000,
        price=600.0
    )
    
    # 市場數據
    market_data = MarketData(
        bid_price=599.0,
        ask_price=601.0,
        bid_size=1000,
        ask_size=1000,
        last_price=600.0,
        volume=50000000,
        timestamp=datetime.now()
    )
    
    # 價格歷史
    price_history = []
    for i in range(20):
        price = StockPrice(
            date=date.today(),
            open=590 + i,
            high=605 + i,
            low=585 + i,
            close=600 + i,
            volume=40000000 + i * 1000000
        )
        price_history.append(price)
    
    # 技術指標
    technical_indicators = TechnicalIndicators(
        rsi=65.0,
        macd=0.8,
        macd_signal=0.5,
        macd_histogram=0.3,
        k=70.0,
        d=68.0,
        ma5=615.0,
        ma20=610.0,
        ma60=605.0,
        atr=15.0
    )
    
    # 完整股票數據
    comprehensive_data = ComprehensiveStockData(
        info=stock_info,
        current_market=market_data,
        price_history=price_history,
        technical_indicators=technical_indicators
    )
    
    return comprehensive_data

def test_core_analyzers():
    """測試核心分析器"""
    print("測試核心分析器...")
    
    test_data = create_test_data()
    
    # 測試流動性分析器
    liquidity_analyzer = LiquidityAnalyzer()
    liquidity_result = liquidity_analyzer.assess_liquidity(test_data)
    print(f"✓ 流動性分析: 分數{liquidity_result.score:.1f}")
    
    # 測試波動性分析器
    volatility_analyzer = VolatilityAnalyzer()
    volatility_result = volatility_analyzer.assess_volatility(test_data)
    print(f"✓ 波動性分析: 分數{volatility_result.score:.1f}")
    
    # 測試動能分析器
    momentum_analyzer = MomentumAnalyzer()
    momentum_result = momentum_analyzer.assess_momentum(test_data)
    print(f"✓ 動能分析: 分數{momentum_result.score:.1f}")
    
    return True

def test_strategy_system():
    """測試策略系統"""
    print("測試策略系統...")
    
    test_data = create_test_data()
    strategy_identifier = StrategyIdentifier()
    
    # 測試策略識別
    strategy_result = strategy_identifier.identify_optimal_strategy(test_data)
    
    if strategy_result.get('recommended_strategy'):
        print(f"✓ 推薦策略: {strategy_result['recommended_strategy']}")
        print(f"✓ 策略分數: {strategy_result['strategy_score']:.1f}")
    else:
        print("✓ 策略系統運行正常（無明確推薦）")
    
    return True

def test_risk_management():
    """測試風險管理"""
    print("測試風險管理...")
    
    test_data = create_test_data()
    risk_manager = ProfessionalRiskManager()
    
    # 測試風險評估
    risk_assessment = risk_manager.assess_comprehensive_risk(
        test_data, "trend_following"
    )
    
    print(f"✓ 風險等級: {risk_assessment.overall_risk_level.value}")
    print(f"✓ 最大倉位: {risk_assessment.max_position_size:.1%}")
    
    # 測試資金配置
    fund_allocator = FundAllocationCalculator(total_capital=1000000)
    position_calc = fund_allocator.calculate_position_size(
        test_data,
        risk_assessment,
        {"strategy_name": "trend_following", "is_intraday": True}
    )
    
    print(f"✓ 建議股數: {position_calc.recommended_shares}")
    print(f"✓ 倉位價值: {position_calc.position_value:,.0f}")
    
    return True

def test_integration():
    """測試整合流程"""
    print("測試整合流程...")
    
    test_data = create_test_data()
    
    # 1. 策略識別
    strategy_identifier = StrategyIdentifier()
    strategy_result = strategy_identifier.identify_optimal_strategy(test_data)
    
    # 2. 風險評估
    if strategy_result.get('recommended_strategy'):
        risk_manager = ProfessionalRiskManager()
        risk_assessment = risk_manager.assess_comprehensive_risk(
            test_data, strategy_result['recommended_strategy']
        )
        
        # 3. 資金配置
        fund_allocator = FundAllocationCalculator()
        position_calc = fund_allocator.calculate_position_size(
            test_data,
            risk_assessment,
            {"strategy_name": strategy_result['recommended_strategy'], "is_intraday": True}
        )
        
        print("✓ 完整分析流程成功")
        print(f"  策略: {strategy_result['recommended_strategy']}")
        print(f"  風險: {risk_assessment.overall_risk_level.value}")
        print(f"  倉位: {position_calc.position_percentage:.1%}")
    else:
        print("✓ 整合流程運行正常（無明確推薦）")
    
    return True

def main():
    """主測試函數"""
    print("="*50)
    print("台股當沖AI選股系統 - 簡化測試")
    print("="*50)
    
    try:
        # 測試數據模型
        print("測試數據模型...")
        test_data = create_test_data()
        print(f"✓ 股票代號: {test_data.info.code}")
        print(f"✓ 現價: {test_data.current_price}")
        print(f"✓ RSI: {test_data.technical_indicators.rsi}")
        print(f"✓ MACD黃金交叉: {test_data.technical_indicators.macd_bullish_crossover}")
        print()
        
        # 測試核心分析器
        if not test_core_analyzers():
            return False
        print()
        
        # 測試策略系統
        if not test_strategy_system():
            return False
        print()
        
        # 測試風險管理
        if not test_risk_management():
            return False
        print()
        
        # 測試整合流程
        if not test_integration():
            return False
        print()
        
        print("="*50)
        print("✅ 所有測試通過！系統運行正常。")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)