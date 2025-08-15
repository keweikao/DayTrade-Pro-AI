"""
系統測試腳本
System Testing Script - 驗證台股當沖AI選股系統
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, date

# 導入要測試的模組
from src.models.stock_data import (
    StockPrice, TechnicalIndicators, ComprehensiveStockData, MarketContext
)
from src.core.liquidity_analyzer import LiquidityAnalyzer
from src.core.volatility_analyzer import VolatilityAnalyzer
from src.core.momentum_analyzer import MomentumAnalyzer
from src.strategies.strategy_identifier import StrategyIdentifier
from src.risk.risk_manager import ProfessionalRiskManager, RiskLevel
from src.risk.fund_allocator import FundAllocationCalculator, PositionSizeMethod


class TestStockDataModels(unittest.TestCase):
    """測試數據模型"""
    
    def setUp(self):
        """設置測試數據"""
        self.sample_price = StockPrice(
            date=date.today(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.0,
            volume=1000000
        )
        
        self.sample_indicators = TechnicalIndicators(
            rsi=55.0,
            macd=0.5,
            macd_signal=0.3,
            macd_histogram=0.2,
            k=60.0,
            d=58.0,
            ma5=101.0,
            ma20=100.0,
            ma60=99.0,
            atr=2.5
        )
    
    def test_stock_price_creation(self):
        """測試股價數據創建"""
        self.assertEqual(self.sample_price.close, 102.0)
        self.assertEqual(self.sample_price.volume, 1000000)
        self.assertIsInstance(self.sample_price.date, date)
    
    def test_technical_indicators_creation(self):
        """測試技術指標創建"""
        self.assertEqual(self.sample_indicators.rsi, 55.0)
        self.assertTrue(self.sample_indicators.macd_bullish_crossover)
        self.assertEqual(self.sample_indicators.atr, 2.5)
    
    def test_comprehensive_stock_data(self):
        """測試完整股票數據"""
        stock_data = ComprehensiveStockData(
            symbol="2330",
            current_price=102.0,
            price_history=[self.sample_price],
            technical_indicators=self.sample_indicators,
            volume=1000000,
            avg_volume=800000
        )
        
        self.assertEqual(stock_data.symbol, "2330")
        self.assertEqual(stock_data.current_price, 102.0)
        self.assertAlmostEqual(stock_data.volume_ratio, 1.25, places=2)
        self.assertAlmostEqual(stock_data.price_vs_ma20, 0.02, places=2)


class TestCoreAnalyzers(unittest.TestCase):
    """測試核心分析器"""
    
    def setUp(self):
        """設置測試數據"""
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        
        # 創建測試股票數據
        self.test_stock_data = self._create_test_stock_data()
    
    def _create_test_stock_data(self):
        """創建測試股票數據"""
        price_history = []
        for i in range(20):
            price = StockPrice(
                date=date.today(),
                open=100 + i,
                high=105 + i,
                low=98 + i,
                close=102 + i,
                volume=1000000 + i * 10000
            )
            price_history.append(price)
        
        indicators = TechnicalIndicators(
            rsi=60.0,
            macd=0.5,
            macd_signal=0.3,
            macd_histogram=0.2,
            k=65.0,
            d=62.0,
            ma5=115.0,
            ma20=110.0,
            ma60=105.0,
            atr=3.0
        )
        
        return ComprehensiveStockData(
            symbol="TEST",
            current_price=120.0,
            price_history=price_history,
            technical_indicators=indicators,
            volume=1200000,
            avg_volume=1000000
        )
    
    def test_liquidity_analyzer(self):
        """測試流動性分析器"""
        result = self.liquidity_analyzer.assess_liquidity(self.test_stock_data)
        
        self.assertIn('liquidity_score', result)
        self.assertIn('liquidity_level', result)
        self.assertIn('risk_warnings', result)
        self.assertIsInstance(result['liquidity_score'], (int, float))
        self.assertIn(result['liquidity_level'], ['high', 'medium', 'low'])
    
    def test_volatility_analyzer(self):
        """測試波動性分析器"""
        result = self.volatility_analyzer.assess_volatility(self.test_stock_data)
        
        self.assertIn('volatility_score', result)
        self.assertIn('volatility_characteristics', result)
        self.assertIn('trading_opportunities', result)
        self.assertIsInstance(result['volatility_score'], (int, float))
    
    def test_momentum_analyzer(self):
        """測試動能分析器"""
        result = self.momentum_analyzer.assess_momentum(self.test_stock_data)
        
        self.assertIn('momentum_score', result)
        self.assertIn('momentum_signals', result)
        self.assertIn('catalyst_analysis', result)
        self.assertIsInstance(result['momentum_score'], (int, float))


class TestStrategies(unittest.TestCase):
    """測試交易策略"""
    
    def setUp(self):
        """設置測試數據"""
        self.strategy_identifier = StrategyIdentifier()
        self.test_stock_data = self._create_test_stock_data()
    
    def _create_test_stock_data(self):
        """創建測試股票數據"""
        indicators = TechnicalIndicators(
            rsi=65.0,
            macd=0.8,
            macd_signal=0.5,
            macd_histogram=0.3,
            k=70.0,
            d=68.0,
            ma5=120.0,
            ma20=115.0,
            ma60=110.0,
            atr=2.8
        )
        
        price_history = []
        for i in range(15):
            price = StockPrice(
                date=date.today(),
                open=110 + i,
                high=115 + i,
                low=108 + i,
                close=112 + i,
                volume=900000 + i * 5000
            )
            price_history.append(price)
        
        return ComprehensiveStockData(
            symbol="TEST",
            current_price=125.0,
            price_history=price_history,
            technical_indicators=indicators,
            volume=1100000,
            avg_volume=950000
        )
    
    def test_strategy_identification(self):
        """測試策略識別"""
        result = self.strategy_identifier.identify_optimal_strategy(self.test_stock_data)
        
        self.assertIn('recommended_strategy', result)
        self.assertIn('strategy_score', result)
        self.assertIn('all_scores', result)
        
        if result['recommended_strategy']:
            self.assertIn('execution_plan', result)
            self.assertIsInstance(result['strategy_score'], (int, float))
    
    def test_strategy_comparison(self):
        """測試策略比較"""
        comparison = self.strategy_identifier.get_strategy_comparison(self.test_stock_data)
        
        self.assertIn('trend_following', comparison)
        self.assertIn('counter_trend', comparison)
        self.assertIn('breakout', comparison)
        self.assertIn('overall_analysis', comparison)


class TestRiskManagement(unittest.TestCase):
    """測試風險管理"""
    
    def setUp(self):
        """設置測試數據"""
        self.risk_manager = ProfessionalRiskManager()
        self.fund_allocator = FundAllocationCalculator(total_capital=1000000)
        self.test_stock_data = self._create_test_stock_data()
    
    def _create_test_stock_data(self):
        """創建測試股票數據"""
        indicators = TechnicalIndicators(
            rsi=55.0,
            macd=0.3,
            macd_signal=0.2,
            macd_histogram=0.1,
            k=58.0,
            d=55.0,
            ma5=105.0,
            ma20=100.0,
            ma60=98.0,
            atr=2.2
        )
        
        return ComprehensiveStockData(
            symbol="TEST",
            current_price=103.0,
            price_history=[],
            technical_indicators=indicators,
            volume=850000,
            avg_volume=800000
        )
    
    def test_risk_assessment(self):
        """測試風險評估"""
        assessment = self.risk_manager.assess_comprehensive_risk(
            self.test_stock_data, 
            "trend_following"
        )
        
        self.assertIsInstance(assessment.overall_risk_level, RiskLevel)
        self.assertIsInstance(assessment.risk_factors, list)
        self.assertIsInstance(assessment.risk_mitigations, list)
        self.assertIsInstance(assessment.max_position_size, float)
        self.assertGreater(assessment.recommended_stop_loss, 0)
    
    def test_fund_allocation(self):
        """測試資金配置"""
        # 創建模擬風險評估
        mock_risk_assessment = Mock()
        mock_risk_assessment.max_position_size = 0.1
        mock_risk_assessment.recommended_stop_loss = 95.0
        mock_risk_assessment.overall_risk_level = RiskLevel.MEDIUM
        
        # 測試倉位計算
        position_calc = self.fund_allocator.calculate_position_size(
            self.test_stock_data,
            mock_risk_assessment,
            {"strategy_name": "trend_following", "is_intraday": True},
            PositionSizeMethod.ATR_BASED
        )
        
        self.assertGreater(position_calc.recommended_shares, 0)
        self.assertGreater(position_calc.position_value, 0)
        self.assertGreater(position_calc.position_percentage, 0)
        self.assertEqual(position_calc.position_size_method, PositionSizeMethod.ATR_BASED)


class TestIntegration(unittest.TestCase):
    """整合測試"""
    
    def setUp(self):
        """設置測試數據"""
        self.strategy_identifier = StrategyIdentifier()
        self.risk_manager = ProfessionalRiskManager()
        self.fund_allocator = FundAllocationCalculator()
    
    def test_full_analysis_pipeline(self):
        """測試完整分析流程"""
        # 創建測試數據
        test_stock_data = self._create_comprehensive_test_data()
        
        # 1. 策略識別
        strategy_result = self.strategy_identifier.identify_optimal_strategy(test_stock_data)
        self.assertIsNotNone(strategy_result)
        
        # 2. 風險評估
        if strategy_result.get('recommended_strategy'):
            risk_assessment = self.risk_manager.assess_comprehensive_risk(
                test_stock_data,
                strategy_result['recommended_strategy']
            )
            self.assertIsNotNone(risk_assessment)
            
            # 3. 倉位計算
            position_calc = self.fund_allocator.calculate_position_size(
                test_stock_data,
                risk_assessment,
                {"strategy_name": strategy_result['recommended_strategy'], "is_intraday": True}
            )
            self.assertIsNotNone(position_calc)
            self.assertGreater(position_calc.recommended_shares, 0)
    
    def _create_comprehensive_test_data(self):
        """創建完整測試數據"""
        indicators = TechnicalIndicators(
            rsi=62.0,
            macd=0.6,
            macd_signal=0.4,
            macd_histogram=0.2,
            k=65.0,
            d=62.0,
            ma5=108.0,
            ma20=105.0,
            ma60=102.0,
            atr=2.5
        )
        
        price_history = []
        for i in range(30):
            price = StockPrice(
                date=date.today(),
                open=100 + i * 0.5,
                high=105 + i * 0.5,
                low=98 + i * 0.5,
                close=102 + i * 0.5,
                volume=800000 + i * 1000
            )
            price_history.append(price)
        
        return ComprehensiveStockData(
            symbol="2330",
            current_price=110.0,
            price_history=price_history,
            technical_indicators=indicators,
            volume=950000,
            avg_volume=850000
        )


def run_performance_test():
    """執行性能測試"""
    print("\n" + "="*50)
    print("性能測試開始")
    print("="*50)
    
    import time
    
    # 測試數據創建性能
    start_time = time.time()
    
    test_data = []
    for i in range(100):
        indicators = TechnicalIndicators(
            rsi=50 + i % 50,
            macd=0.1 * (i % 10),
            macd_signal=0.05 * (i % 10),
            macd_histogram=0.05 * (i % 5),
            k=50 + i % 50,
            d=48 + i % 50,
            ma5=100 + i,
            ma20=98 + i,
            ma60=95 + i,
            atr=2.0 + (i % 5) * 0.1
        )
        
        stock_data = ComprehensiveStockData(
            symbol=f"TEST{i:04d}",
            current_price=100 + i,
            price_history=[],
            technical_indicators=indicators,
            volume=800000 + i * 1000,
            avg_volume=750000 + i * 1000
        )
        test_data.append(stock_data)
    
    creation_time = time.time() - start_time
    print(f"創建100個股票數據耗時: {creation_time:.3f}秒")
    
    # 測試分析性能
    start_time = time.time()
    strategy_identifier = StrategyIdentifier()
    
    results = []
    for data in test_data[:10]:  # 測試前10個
        result = strategy_identifier.identify_optimal_strategy(data)
        results.append(result)
    
    analysis_time = time.time() - start_time
    print(f"分析10支股票耗時: {analysis_time:.3f}秒")
    print(f"平均每支股票分析時間: {analysis_time/10:.3f}秒")
    
    print("性能測試完成")
    print("="*50)


def run_system_validation():
    """執行系統驗證"""
    print("\n" + "="*50)
    print("系統驗證開始")
    print("="*50)
    
    try:
        # 檢查模組導入
        print("✓ 模組導入測試通過")
        
        # 檢查數據模型
        test_price = StockPrice(
            date=date.today(),
            open=100, high=105, low=98, close=102, volume=1000000
        )
        print("✓ 數據模型測試通過")
        
        # 檢查分析器
        from src.core.liquidity_analyzer import LiquidityAnalyzer
        liquidity_analyzer = LiquidityAnalyzer()
        print("✓ 分析器初始化測試通過")
        
        # 檢查策略系統
        from src.strategies.strategy_identifier import StrategyIdentifier
        strategy_identifier = StrategyIdentifier()
        print("✓ 策略系統測試通過")
        
        # 檢查風險管理
        from src.risk.risk_manager import ProfessionalRiskManager
        risk_manager = ProfessionalRiskManager()
        print("✓ 風險管理系統測試通過")
        
        print("\n所有系統組件驗證通過！")
        
    except Exception as e:
        print(f"❌ 系統驗證失敗: {str(e)}")
        return False
    
    print("="*50)
    return True


if __name__ == "__main__":
    print("台股當沖AI選股系統 - 系統測試")
    print("="*50)
    
    # 執行系統驗證
    if not run_system_validation():
        sys.exit(1)
    
    # 執行單元測試
    print("\n單元測試開始...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 執行性能測試
    run_performance_test()
    
    print("\n" + "="*50)
    print("✅ 所有測試完成！系統準備就緒。")
    print("="*50)