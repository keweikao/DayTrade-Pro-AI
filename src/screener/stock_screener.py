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
            # 權值股 (台積電、鴻海、聯發科等)
            "2330", "2317", "2454", "2308", "3008", "1303", "1301", "2382", "2395", "2412",
            
            # 半導體股
            "2330", "2454", "3711", "2379", "3034", "2401", "2409", "3443", "8046", "2388",
            
            # 電子製造
            "2317", "2382", "2395", "3231", "2324", "2357", "4938", "6505", "3045", "2377",
            
            # 金融股
            "2880", "2882", "2884", "2885", "2886", "2887", "2890", "2891", "2892", "5880",
            "2883", "2801", "2809", "2834", "2845", "2849", "2856", "2888", "1216", "2327",
            
            # 傳統產業
            "1301", "1303", "1326", "2002", "2105", "1102", "2207", "1216", "2408", "1605",
            "2201", "2301", "2609", "1101", "2912", "3481", "1904", "4904", "2633", "9910",
            
            # 生技醫療
            "4562", "6446", "4306", "6239", "1560", "1477", "6491", "4174", "1809", "6492",
            "4133", "3293", "6547", "4120", "6547", "3596", "1777", "4154", "6532", "8422",
            
            # 航運海運
            "2603", "2609", "2615", "2606", "5880", "2610", "2634", "2636", "2645", "1603",
            
            # 電信通訊
            "3045", "2412", "3006", "4904", "6176", "3533", "3481", "2351", "8086", "2349",
            
            # 觀光餐飲
            "2712", "5522", "2705", "2724", "1229", "2748", "5903", "2231", "2729", "2707",
            
            # 營建房地產
            "5522", "9945", "5871", "2548", "5425", "1909", "2542", "2515", "1537", "2540",
            
            # 汽車零件
            "2207", "2201", "2204", "1513", "6116", "2809", "2230", "2236", "2227", "2228",
            
            # 化工塑膠
            "1301", "1303", "1326", "4737", "1722", "4736", "4163", "1802", "1712", "1717",
            
            # 熱門ETF
            "0050", "0056", "006208", "00878", "00713", "00881", "00900", "00692", "00757", "00919"
        ]
        
        # 移除重複股票代號
        self.stock_universe = list(set(self.stock_universe))
    
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
            
            strategy_score = strategy_result.get('strategy_score', 0)
            print(f"   📊 {stock_data.symbol} 策略評分: {strategy_score:.1f} (要求: {min_score})")
            
            if strategy_score >= min_score:
                analyzed_stocks.append({
                    'stock_data': stock_data,
                    'strategy_result': strategy_result
                })
                print(f"   ✅ {stock_data.symbol} 通過策略篩選")
            else:
                print(f"   ❌ {stock_data.symbol} 策略評分不足")
        
        print(f"🎯 策略分析完成，符合條件 {len(analyzed_stocks)} 支股票")
        
        # 如果策略分析後沒有結果，降低評分要求
        if len(analyzed_stocks) == 0 and len(filtered_stocks) > 0:
            lowered_score = max(30.0, min_score - 20)  # 降低20分，但不低於30分
            print(f"⚠️ 策略篩選無結果，降低要求至 {lowered_score} 分...")
            
            for i, stock_data in enumerate(filtered_stocks):
                strategy_result = self.strategy_identifier.identify_optimal_strategy(stock_data, market_context)
                strategy_score = strategy_result.get('strategy_score', 0)
                
                if strategy_score >= lowered_score:
                    analyzed_stocks.append({
                        'stock_data': stock_data,
                        'strategy_result': strategy_result
                    })
                    print(f"   ✅ {stock_data.symbol} 通過降低標準 (評分: {strategy_score:.1f})")
                    
                    if len(analyzed_stocks) >= max_recommendations:
                        break
        
        # 最後保險：如果還是沒有結果，直接使用前幾支股票
        if len(analyzed_stocks) == 0 and len(filtered_stocks) > 0:
            print("⚠️ 所有策略標準都無結果，直接使用基本篩選結果...")
            for stock_data in filtered_stocks[:max_recommendations]:
                # 創建一個基本的策略結果
                basic_strategy = {
                    'recommended_strategy': 'basic_analysis',
                    'strategy_score': 50.0,  # 給一個中性分數
                    'confidence_level': 0.5,
                    'all_scores': {'basic': 50.0}
                }
                analyzed_stocks.append({
                    'stock_data': stock_data,
                    'strategy_result': basic_strategy
                })
        
        print(f"🎯 最終策略分析: {len(analyzed_stocks)} 支股票")
        
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
        all_valid_stocks = []  # 保存所有能獲取數據的股票
        
        print(f"🔍 開始檢查 {len(self.stock_universe)} 支股票...")
        
        for i, symbol in enumerate(self.stock_universe):
            print(f"📈 檢查進度: {i+1}/{len(self.stock_universe)} - {symbol}")
            try:
                # 獲取股票數據
                stock_data = self.data_provider.get_comprehensive_stock_data(symbol)
                
                if stock_data is None:
                    print(f"   ❌ {symbol} 無法獲取數據")
                    continue
                
                all_valid_stocks.append(stock_data)
                
                # 先嘗試嚴格標準
                if self._meets_basic_criteria(stock_data, strict_mode=True):
                    filtered_stocks.append(stock_data)
                    
            except Exception as e:
                print(f"   ⚠️ 獲取 {symbol} 數據失敗: {e}")
                continue
        
        # 如果嚴格標準沒有結果，使用寬鬆標準
        if len(filtered_stocks) == 0 and len(all_valid_stocks) > 0:
            print("⚠️ 嚴格標準無結果，改用寬鬆標準...")
            for stock_data in all_valid_stocks:
                if self._meets_basic_criteria(stock_data, strict_mode=False):
                    filtered_stocks.append(stock_data)
        
        # 最後的保險：如果還是沒有結果，至少返回一些有效的股票
        if len(filtered_stocks) == 0 and len(all_valid_stocks) > 0:
            print("⚠️ 所有標準都無結果，返回前5支有效股票作為演示...")
            filtered_stocks = all_valid_stocks[:5]
        
        print(f"📊 基本篩選完成: {len(filtered_stocks)}/{len(self.stock_universe)} 支股票通過")
        return filtered_stocks
    
    def _meets_basic_criteria(self, stock_data, strict_mode: bool = False) -> bool:
        """檢查是否符合基本當沖條件"""
        
        try:
            symbol = stock_data.symbol
            
            # 設定不同的標準
            if strict_mode:
                min_volume_ratio = 0.8
                min_atr_pct = 0.015
                min_price, max_price = 10, 1000
                min_rsi, max_rsi = 10, 90
            else:
                # 寬鬆標準 - 用於確保有結果
                min_volume_ratio = 0.3
                min_atr_pct = 0.005
                min_price, max_price = 5, 2000
                min_rsi, max_rsi = 5, 95
            
            # 1. 流動性檢查 - 成交量足夠
            volume_ratio = stock_data.volume_ratio
            if volume_ratio < min_volume_ratio:
                print(f"   ❌ {symbol} 量比不足: {volume_ratio:.2f} < {min_volume_ratio}")
                return False
            
            # 2. 波動性檢查 - 有足夠的價格變動
            atr_pct = stock_data.atr_percentage
            if atr_pct < min_atr_pct:
                print(f"   ❌ {symbol} 波動性不足: {atr_pct:.3f} < {min_atr_pct}")
                return False
            
            # 3. 價格檢查 - 避免過低或過高的股票
            current_price = stock_data.current_price
            if current_price < min_price or current_price > max_price:
                print(f"   ❌ {symbol} 價格超出範圍: {current_price} (範圍:{min_price}-{max_price})")
                return False
            
            # 4. 技術面檢查 - 避免極端情況
            rsi = stock_data.technical_indicators.rsi
            if rsi < min_rsi or rsi > max_rsi:
                print(f"   ❌ {symbol} RSI極端: {rsi:.1f} (範圍:{min_rsi}-{max_rsi})")
                return False
            
            mode_text = "嚴格" if strict_mode else "寬鬆"
            print(f"   ✅ {symbol} 通過{mode_text}篩選 (量比:{volume_ratio:.2f}, ATR:{atr_pct:.3f}, 價格:{current_price:.2f}, RSI:{rsi:.1f})")
            return True
            
        except Exception as e:
            print(f"   ❌ {stock_data.symbol if hasattr(stock_data, 'symbol') else 'Unknown'} 檢查失敗: {e}")
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
        
        # 動態計算各類別股票數量
        categories_samples = {
            "權值股": ["2330台積電", "2317鴻海", "2454聯發科", "2308台達電", "3008大立光"],
            "半導體": ["2330台積電", "2454聯發科", "3711日月光", "2379瑞昱", "3034聯詠"],
            "電子製造": ["2317鴻海", "2382廣達", "2395研華", "3231緯創", "2324仁寶"],
            "金融股": ["2880華南金", "2882國泰金", "2884玉山金", "2885元大金", "2886兆豐金"],
            "傳統產業": ["1301台塑", "1303南亞", "2002中鋼", "2105正新", "1102亞泥"],
            "生技醫療": ["4562偉聯", "6446藥華藥", "4306炎洲", "6239力成", "1560中砂"],
            "航運海運": ["2603長榮", "2609陽明", "2615萬海", "2606裕民", "2610華航"],
            "熱門ETF": ["0050台灣50", "0056高股息", "006208富邦台50", "00878國泰永續", "00713元大台灣高息低波"]
        }
        
        return {
            "total_stocks": len(self.stock_universe),
            "categories": categories_samples,
            "screening_criteria": {
                "成交量比率": "> 0.8 (嚴格) / > 0.3 (寬鬆)",
                "平均真實波幅": "> 1.5% (嚴格) / > 0.5% (寬鬆)",
                "股價範圍": "10-1000元 (嚴格) / 5-2000元 (寬鬆)",
                "RSI範圍": "10-90 (嚴格) / 5-95 (寬鬆)"
            },
            "data_sources": [
                "台灣證券交易所 (上市)",
                "證券櫃檯買賣中心 (上櫃)", 
                "Yahoo Finance API",
                "即時技術指標計算"
            ]
        }
    
    def update_stock_universe(self, new_stocks: List[str]):
        """更新股票池"""
        self.stock_universe.extend(new_stocks)
        self.stock_universe = list(set(self.stock_universe))  # 去重
    
    def add_stocks_by_category(self, category: str) -> int:
        """按類別添加更多股票"""
        
        additional_stocks = {
            "大型權值股": [
                "2330", "2317", "2454", "2308", "3008", "2382", "2395", "2412", "1303", "1301",
                "2002", "2886", "2880", "2882", "6505", "3711", "2891", "2324", "3045", "2357"
            ],
            "中小型成長股": [
                "3034", "2379", "3443", "8046", "2388", "2409", "2401", "3231", "2377", "4938",
                "6176", "3533", "6415", "3596", "4966", "5347", "6147", "3406", "6281", "4904"
            ],
            "金融完整版": [
                "2880", "2882", "2884", "2885", "2886", "2887", "2890", "2891", "2892", "5880",
                "2883", "2801", "2809", "2834", "2845", "2849", "2856", "2888", "1216", "2327",
                "5820", "9958", "2816", "2823", "2832", "2836", "2838", "2867", "9907", "9921"
            ],
            "生技醫療全": [
                "4562", "6446", "4306", "6239", "1560", "1477", "6491", "4174", "1809", "6492",
                "4133", "3293", "6547", "4120", "3596", "1777", "4154", "6532", "8422", "4152",
                "1762", "4745", "6469", "6535", "4142", "4736", "4168", "4114", "1789", "4119"
            ],
            "航運海運全": [
                "2603", "2609", "2615", "2606", "5880", "2610", "2634", "2636", "2645", "1603",
                "2618", "2637", "2612", "5007", "6016", "4426", "2617", "2642", "5608", "6691"
            ],
            "熱門ETF": [
                "0050", "0056", "006208", "00878", "00713", "00881", "00900", "00692", "00757", "00919",
                "00929", "00934", "00940", "00944", "00893", "00895", "00896", "00897", "00888", "00891"
            ]
        }
        
        if category in additional_stocks:
            new_stocks = additional_stocks[category]
            original_count = len(self.stock_universe)
            self.update_stock_universe(new_stocks)
            added_count = len(self.stock_universe) - original_count
            print(f"✅ 已添加 {category}，新增 {added_count} 支股票，總計 {len(self.stock_universe)} 支")
            return added_count
        else:
            available_categories = list(additional_stocks.keys())
            print(f"❌ 不支援的類別：{category}")
            print(f"可用類別：{', '.join(available_categories)}")
            return 0
    
    def load_taiwan_top_stocks(self, top_n: int = 100):
        """載入台灣市值前N大股票 (模擬實現)"""
        
        # 台灣市值前100大股票代號 (簡化版，實際應從交易所API獲取)
        top_100_stocks = [
            # 市值前20大
            "2330", "2317", "2454", "2308", "3008", "2382", "2395", "1303", "1301", "2002",
            "2412", "2886", "2880", "2882", "6505", "3711", "2891", "2324", "3045", "2357",
            
            # 市值21-50大
            "2379", "3034", "2409", "2401", "3443", "8046", "2388", "3231", "2377", "4938",
            "6176", "3533", "6415", "3596", "4966", "5347", "6147", "3406", "6281", "4904",
            "2884", "2885", "2887", "2890", "2892", "5880", "2883", "2801", "2809", "2834",
            
            # 市值51-80大
            "2845", "2849", "2856", "2888", "1216", "2327", "5820", "9958", "2816", "2823",
            "2832", "2836", "2838", "2867", "9907", "9921", "2105", "1102", "2207", "2408",
            "1605", "2201", "2301", "2609", "1101", "2912", "3481", "1904", "4904", "2633",
            
            # 市值81-100大
            "9910", "2603", "2615", "2606", "2610", "2634", "2636", "2645", "1603", "4562",
            "6446", "4306", "6239", "1560", "1477", "6491", "4174", "1809", "6492", "4133"
        ]
        
        stocks_to_add = top_100_stocks[:top_n]
        original_count = len(self.stock_universe)
        self.update_stock_universe(stocks_to_add)
        added_count = len(self.stock_universe) - original_count
        
        print(f"✅ 已載入台股市值前 {top_n} 大股票，新增 {added_count} 支，總計 {len(self.stock_universe)} 支")
        return added_count