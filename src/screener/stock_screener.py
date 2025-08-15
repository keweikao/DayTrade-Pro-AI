"""
智能選股篩選器
Intelligent Stock Screener - 自動推薦適合當沖的股票
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

from ..data.taiwan_market_data import TaiwanMarketDataProvider
from ..strategies.strategy_identifier import StrategyIdentifier
from ..risk.risk_manager import ProfessionalRiskManager
from ..ai.stock_analyzer import AIStockAnalyzer


@dataclass
class StockRecommendation:
    """股票推薦"""
    symbol: str
    name: str
    current_price: float
    previous_close: float  # 昨日收盤價
    recommendation_score: float
    recommended_strategy: str
    ai_analysis: Any
    risk_level: str
    entry_price: float
    exit_price: float  # 建議出場價格
    target_price: float  # 目標價格
    stop_loss: float
    reason: str
    confidence: float
    # 新增分析數據
    price_change: float  # 今日漲跌
    price_change_pct: float  # 今日漲跌幅
    volume: int  # 成交量
    volume_ratio: float  # 量比
    turnover_rate: float  # 週轉率


class IntelligentStockScreener:
    """智能股票篩選器"""
    
    def __init__(self, openai_api_key: str = None):
        self.data_provider = TaiwanMarketDataProvider()
        self.strategy_identifier = StrategyIdentifier()
        self.risk_manager = ProfessionalRiskManager()
        self.ai_analyzer = AIStockAnalyzer(openai_api_key) if openai_api_key else None
        
        # 台股熱門當沖標的池（可擴展）
        self.stock_universe = [
            # 權值股
            "2330", "2454", "2317", "3034", "2882", "1303", "1301",
            # 電子股
            "2412", "2891", "2886", "6505", "3711", "4938", "2357",
            # 金融股  
            "2880", "2884", "2885", "2892", "5880", "2883",
            # 傳產股
            "1216", "2207", "1326", "2002", "2105", "1102",
            # 生技股
            "4562", "6446", "4306", "6239", "1560", "1477",
            # 航運股
            "2603", "2609", "2615", "5880", "2606"
        ]
    
    async def get_daily_recommendations(self, 
                                      max_recommendations: int = 10,
                                      min_score: float = 70.0) -> List[StockRecommendation]:
        """
        獲取每日推薦股票
        
        Args:
            max_recommendations: 最大推薦數量
            min_score: 最低推薦分數
            
        Returns:
            List[StockRecommendation]: 推薦股票列表
        """
        
        print("🔍 開始掃描台股市場...")
        
        # 1. 初步篩選 - 快速過濾基本條件
        filtered_stocks = await self._pre_filter_stocks()
        print(f"📊 初步篩選完成，剩餘 {len(filtered_stocks)} 支股票")
        
        # 2. 詳細分析 - 策略適用性評估
        analyzed_stocks = []
        market_context = self.data_provider.get_market_context()
        
        for i, stock_data in enumerate(filtered_stocks):
            print(f"📈 分析進度: {i+1}/{len(filtered_stocks)} - {stock_data.symbol}")
            
            # 策略分析
            strategy_result = self.strategy_identifier.identify_optimal_strategy(stock_data, market_context)
            
            if strategy_result.get('strategy_score', 0) >= min_score:
                analyzed_stocks.append({
                    'stock_data': stock_data,
                    'strategy_result': strategy_result
                })
        
        print(f"🎯 策略分析完成，符合條件 {len(analyzed_stocks)} 支股票")
        
        # 3. AI深度分析（如果有API Key）
        if self.ai_analyzer and analyzed_stocks:
            print("🤖 開始AI深度分析...")
            recommendations = await self._ai_analysis_batch(analyzed_stocks[:max_recommendations])
        else:
            print("📝 生成基礎推薦...")
            recommendations = self._generate_basic_recommendations(analyzed_stocks[:max_recommendations])
        
        # 4. 排序並返回最終推薦
        recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
        
        print(f"✅ 推薦生成完成，共 {len(recommendations)} 支股票")
        return recommendations[:max_recommendations]
    
    async def _pre_filter_stocks(self) -> List[Any]:
        """初步篩選股票 - 基於基本條件"""
        
        filtered_stocks = []
        
        for symbol in self.stock_universe:
            try:
                # 獲取股票數據
                stock_data = self.data_provider.get_comprehensive_stock_data(symbol)
                
                if stock_data and self._meets_basic_criteria(stock_data):
                    filtered_stocks.append(stock_data)
                    
            except Exception as e:
                print(f"⚠️ 獲取 {symbol} 數據失敗: {e}")
                continue
        
        return filtered_stocks
    
    def _meets_basic_criteria(self, stock_data) -> bool:
        """檢查是否符合基本當沖條件"""
        
        try:
            # 1. 流動性檢查 - 成交量足夠
            if stock_data.volume_ratio < 0.8:  # 成交量至少是平均的80%
                return False
            
            # 2. 波動性檢查 - 有足夠的價格變動
            if stock_data.atr_percentage < 0.015:  # ATR至少1.5%
                return False
            
            # 3. 價格檢查 - 避免過低或過高的股票
            if stock_data.current_price < 10 or stock_data.current_price > 1000:
                return False
            
            # 4. 技術面檢查 - 避免極端情況
            rsi = stock_data.technical_indicators.rsi
            if rsi < 10 or rsi > 90:  # 避免極端超買超賣
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _ai_analysis_batch(self, analyzed_stocks: List[Dict]) -> List[StockRecommendation]:
        """批量AI分析"""
        
        recommendations = []
        
        # 控制併行數量避免API限制
        semaphore = asyncio.Semaphore(3)
        
        async def analyze_single(stock_info):
            async with semaphore:
                try:
                    stock_data = stock_info['stock_data']
                    strategy_result = stock_info['strategy_result']
                    
                    # 獲取市場環境數據
                    market_context = self.data_provider.get_market_context()
                    
                    # AI分析
                    ai_result = await self.ai_analyzer.analyze_stock_comprehensive(stock_data, market_context)
                    
                    # 風險評估
                    risk_assessment = self.risk_manager.assess_comprehensive_risk(
                        stock_data, strategy_result.get('recommended_strategy', ''), market_context
                    )
                    
                    # 生成推薦
                    recommendation = self._create_recommendation(
                        stock_data, strategy_result, ai_result, risk_assessment
                    )
                    
                    return recommendation
                    
                except Exception as e:
                    print(f"❌ AI分析失敗 {stock_info['stock_data'].symbol}: {e}")
                    return None
        
        # 執行批量分析
        tasks = [analyze_single(stock_info) for stock_info in analyzed_stocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 過濾有效結果
        recommendations = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        return recommendations
    
    def _generate_basic_recommendations(self, analyzed_stocks: List[Dict]) -> List[StockRecommendation]:
        """生成基礎推薦（無AI分析）"""
        
        recommendations = []
        market_context = self.data_provider.get_market_context()
        
        for stock_info in analyzed_stocks:
            try:
                stock_data = stock_info['stock_data']
                strategy_result = stock_info['strategy_result']
                
                # 基礎風險評估
                risk_assessment = self.risk_manager.assess_comprehensive_risk(
                    stock_data, strategy_result.get('recommended_strategy', ''), market_context
                )
                
                # 生成推薦
                recommendation = self._create_recommendation(
                    stock_data, strategy_result, None, risk_assessment
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                print(f"❌ 基礎分析失敗 {stock_data.symbol}: {e}")
                continue
        
        return recommendations
    
    def _create_recommendation(self, stock_data, strategy_result, ai_result, risk_assessment) -> StockRecommendation:
        """創建股票推薦"""
        
        current_price = stock_data.current_price
        atr = stock_data.technical_indicators.atr
        
        # 獲取昨日收盤價
        previous_close = stock_data.price_history[-2].close if len(stock_data.price_history) >= 2 else current_price
        
        # 計算今日漲跌
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100 if previous_close > 0 else 0
        
        # 計算進場價、出場價、目標價和停損價
        if strategy_result.get('recommended_strategy') == 'trend_following':
            # 趨勢跟隨策略
            entry_price = current_price * 1.002  # 略高於現價進場
            exit_price = current_price + (atr * 1.8)  # 保守出場點
            target_price = current_price + (atr * 2.5)  # 積極目標價
        elif strategy_result.get('recommended_strategy') == 'counter_trend':
            # 逆勢策略
            entry_price = current_price * 0.998  # 略低於現價進場
            exit_price = current_price + (atr * 1.2)  # 保守出場點
            target_price = current_price + (atr * 1.8)  # 保守目標價
        else:  # breakout
            # 突破策略
            entry_price = current_price * 1.005  # 突破確認後進場
            exit_price = current_price + (atr * 2.0)  # 標準出場點
            target_price = current_price + (atr * 3.0)  # 積極目標價
        
        stop_loss = risk_assessment.recommended_stop_loss
        
        # 綜合評分
        strategy_score = strategy_result.get('strategy_score', 0)
        ai_score = ai_result.overall_score if ai_result else strategy_score
        final_score = (strategy_score + ai_score) / 2
        
        # 推薦理由
        if ai_result:
            reason = ai_result.key_insights[0] if ai_result.key_insights else strategy_result.get('recommended_strategy', '')
            confidence = ai_result.confidence_level
        else:
            reason = f"{strategy_result.get('recommended_strategy', '')}策略適用"
            confidence = 0.7
        
        return StockRecommendation(
            symbol=stock_data.symbol,
            name=stock_data.info.name,
            current_price=current_price,
            previous_close=previous_close,
            recommendation_score=final_score,
            recommended_strategy=strategy_result.get('recommended_strategy', ''),
            ai_analysis=ai_result,
            risk_level=risk_assessment.overall_risk_level.value,
            entry_price=entry_price,
            exit_price=exit_price,
            target_price=target_price,
            stop_loss=stop_loss,
            reason=reason,
            confidence=confidence,
            price_change=price_change,
            price_change_pct=price_change_pct,
            volume=stock_data.current_market.volume,
            volume_ratio=stock_data.volume_ratio,
            turnover_rate=stock_data.turnover_rate
        )
    
    def get_stock_universe_info(self) -> Dict[str, Any]:
        """獲取股票池資訊"""
        
        return {
            "total_stocks": len(self.stock_universe),
            "categories": {
                "權值股": ["2330", "2454", "2317", "3034", "2882", "1303", "1301"],
                "電子股": ["2412", "2891", "2886", "6505", "3711", "4938", "2357"],
                "金融股": ["2880", "2884", "2885", "2892", "5880", "2883"],
                "傳產股": ["1216", "2207", "1326", "2002", "2105", "1102"],
                "生技股": ["4562", "6446", "4306", "6239", "1560", "1477"],
                "航運股": ["2603", "2609", "2615", "5880", "2606"]
            },
            "screening_criteria": {
                "成交量比率": "> 0.8",
                "平均真實波幅": "> 1.5%",
                "股價範圍": "10-1000元",
                "RSI範圍": "10-90"
            }
        }
    
    def update_stock_universe(self, new_stocks: List[str]):
        """更新股票池"""
        self.stock_universe.extend(new_stocks)
        self.stock_universe = list(set(self.stock_universe))  # 去重