"""
AI股票分析引擎
AI Stock Analysis Engine - 基於OpenAI GPT-4的智能分析
"""
from typing import Dict, List, Any, Optional
import json
import openai
from dataclasses import dataclass
import asyncio

from ..models.stock_data import ComprehensiveStockData, MarketContext
from ..strategies import StrategyIdentifier
from ..risk import ProfessionalRiskManager, FundAllocationCalculator


@dataclass 
class AIAnalysisResult:
    """AI分析結果"""
    stock_symbol: str
    overall_score: float
    recommendation: str
    key_insights: List[str]
    risk_warnings: List[str]
    strategy_advice: str
    entry_timing: str
    confidence_level: float
    reasoning: str


class AIStockAnalyzer:
    """AI股票分析引擎"""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.strategy_identifier = StrategyIdentifier()
        self.risk_manager = ProfessionalRiskManager()
        
        # AI分析參數
        self.model = "gpt-4"
        self.temperature = 0.1  # 保持分析客觀
        self.max_tokens = 2000
        
        # 分析框架
        self.analysis_framework = {
            "technical_analysis": {
                "weight": 0.4,
                "factors": ["趨勢", "支撐阻力", "技術指標", "成交量", "型態"]
            },
            "risk_assessment": {
                "weight": 0.3, 
                "factors": ["流動性", "波動性", "市場風險", "個股風險", "時間風險"]
            },
            "strategy_fit": {
                "weight": 0.2,
                "factors": ["策略適用性", "執行可行性", "風險回報比", "勝率評估"]
            },
            "market_context": {
                "weight": 0.1,
                "factors": ["大盤環境", "行業趨勢", "市場情緒", "時段特性"]
            }
        }
    
    async def analyze_stock_comprehensive(self, 
                                        stock_data: ComprehensiveStockData,
                                        market_context: MarketContext = None) -> AIAnalysisResult:
        """
        全面AI股票分析
        
        Args:
            stock_data: 完整股票數據
            market_context: 市場環境數據
            
        Returns:
            AIAnalysisResult: AI分析結果
        """
        
        # 1. 執行基礎分析
        strategy_result = self.strategy_identifier.identify_optimal_strategy(
            stock_data, market_context
        )
        
        risk_assessment = self.risk_manager.assess_comprehensive_risk(
            stock_data, strategy_result.get('recommended_strategy', ''), market_context
        )
        
        # 2. 準備AI分析提示
        analysis_prompt = self._prepare_analysis_prompt(
            stock_data, strategy_result, risk_assessment, market_context
        )
        
        # 3. 調用AI進行深度分析
        ai_response = await self._call_openai_analysis(analysis_prompt)
        
        # 4. 整合分析結果
        ai_result = self._process_ai_response(ai_response, stock_data.symbol)
        
        return ai_result
    
    def _prepare_analysis_prompt(self, 
                               stock_data: ComprehensiveStockData,
                               strategy_result: Dict[str, Any],
                               risk_assessment,
                               market_context: MarketContext = None) -> str:
        """準備AI分析提示"""
        
        # 構建股票基本資訊
        stock_info = {
            "symbol": stock_data.symbol,
            "current_price": stock_data.current_price,
            "price_vs_ma20": f"{stock_data.price_vs_ma20:.2%}",
            "volume_ratio": f"{stock_data.volume_ratio:.2f}",
            "atr_percentage": f"{stock_data.atr_percentage:.2%}",
            "turnover_rate": f"{stock_data.turnover_rate:.3%}"
        }
        
        # 技術指標摘要
        indicators = stock_data.technical_indicators
        technical_summary = {
            "rsi": f"{indicators.rsi:.1f}",
            "macd": f"{indicators.macd:.4f}",
            "macd_histogram": f"{indicators.macd_histogram:.4f}",
            "k": f"{indicators.k:.1f}",
            "d": f"{indicators.d:.1f}",
            "ma_trend": stock_data.ma_trend
        }
        
        # 策略分析摘要
        strategy_summary = {}
        if strategy_result.get('recommended_strategy'):
            strategy_summary = {
                "recommended_strategy": strategy_result['recommended_strategy'],
                "strategy_score": strategy_result['strategy_score'],
                "confidence_level": strategy_result['confidence_level'],
                "all_scores": strategy_result['all_scores']
            }
        
        # 風險評估摘要
        risk_summary = {
            "overall_risk_level": risk_assessment.overall_risk_level.value,
            "max_position_size": f"{risk_assessment.max_position_size:.1%}",
            "risk_factors_count": len(risk_assessment.risk_factors),
            "special_warnings_count": len(risk_assessment.special_warnings)
        }
        
        # 市場環境
        market_summary = {}
        if market_context:
            market_summary = {
                "is_market_bullish": market_context.is_market_bullish,
                "volatility_environment": market_context.volatility_environment,
                "trading_session": market_context.trading_session
            }
        
        prompt = f"""
你是一位專業的台灣股市當沖交易分析師，具備豐富的技術分析和風險管理經驗。
請基於以下數據，對股票 {stock_data.symbol} 進行全面的當沖交易分析。

## 股票基本資訊
{json.dumps(stock_info, ensure_ascii=False, indent=2)}

## 技術指標分析
{json.dumps(technical_summary, ensure_ascii=False, indent=2)}

## 策略分析結果
{json.dumps(strategy_summary, ensure_ascii=False, indent=2)}

## 風險評估摘要
{json.dumps(risk_summary, ensure_ascii=False, indent=2)}

## 市場環境
{json.dumps(market_summary, ensure_ascii=False, indent=2)}

請根據專業交易員的視角，提供以下分析：

1. **整體評分** (0-100分): 當沖交易適合度
2. **投資建議** (強烈推薦/推薦/中性/謹慎/避免): 明確的操作建議
3. **關鍵洞察** (3-5點): 最重要的分析發現
4. **風險警示** (2-4點): 需要特別注意的風險點
5. **策略建議**: 具體的交易策略建議
6. **進場時機**: 最佳進場時點建議
7. **信心度** (0-1): 對分析結果的信心程度
8. **分析邏輯**: 詳細的推理過程

請以JSON格式回應，格式如下：
{{
    "overall_score": 分數,
    "recommendation": "建議",
    "key_insights": ["洞察1", "洞察2", ...],
    "risk_warnings": ["警示1", "警示2", ...],
    "strategy_advice": "策略建議",
    "entry_timing": "時機建議", 
    "confidence_level": 信心度,
    "reasoning": "分析邏輯"
}}

注意：
- 分析需基於台灣股市特性（09:00-13:30交易時間，±10%漲跌幅限制）
- 重點關注當日交易機會，而非長期投資
- 風險控制是第一優先
- 提供具體可操作的建議
"""
        
        return prompt
    
    async def _call_openai_analysis(self, prompt: str) -> str:
        """調用OpenAI API進行分析"""
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是專業的台灣股市當沖交易分析師，具備深厚的技術分析和風險管理經驗。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # 如果AI調用失敗，返回基礎分析
            return self._generate_fallback_analysis()
    
    def _process_ai_response(self, ai_response: str, symbol: str) -> AIAnalysisResult:
        """處理AI回應並結構化"""
        
        try:
            # 嘗試解析JSON回應
            ai_data = json.loads(ai_response)
            
            return AIAnalysisResult(
                stock_symbol=symbol,
                overall_score=float(ai_data.get('overall_score', 50)),
                recommendation=ai_data.get('recommendation', '中性'),
                key_insights=ai_data.get('key_insights', []),
                risk_warnings=ai_data.get('risk_warnings', []),
                strategy_advice=ai_data.get('strategy_advice', ''),
                entry_timing=ai_data.get('entry_timing', ''),
                confidence_level=float(ai_data.get('confidence_level', 0.5)),
                reasoning=ai_data.get('reasoning', '')
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # 如果解析失敗，創建默認結果
            return AIAnalysisResult(
                stock_symbol=symbol,
                overall_score=50,
                recommendation='中性',
                key_insights=['AI分析暫時不可用'],
                risk_warnings=['請人工複核分析結果'],
                strategy_advice='建議觀望等待更多資訊',
                entry_timing='暫緩操作',
                confidence_level=0.3,
                reasoning='AI分析處理異常，建議人工分析'
            )
    
    def _generate_fallback_analysis(self) -> str:
        """生成備用分析（當AI不可用時）"""
        
        fallback_result = {
            "overall_score": 50,
            "recommendation": "中性",
            "key_insights": [
                "AI分析服務暫時不可用",
                "建議基於技術指標進行人工分析",
                "謹慎評估風險後再做決定"
            ],
            "risk_warnings": [
                "AI分析不可用，增加人工審查",
                "市場變化快速，密切關注"
            ],
            "strategy_advice": "等待AI服務恢復或進行人工分析",
            "entry_timing": "建議暫緩，等待更多資訊",
            "confidence_level": 0.3,
            "reasoning": "AI分析服務不可用，建議謹慎操作並增加人工分析比重"
        }
        
        return json.dumps(fallback_result, ensure_ascii=False)
    
    def analyze_multiple_stocks(self, 
                              stock_list: List[ComprehensiveStockData],
                              market_context: MarketContext = None,
                              max_concurrent: int = 3) -> List[AIAnalysisResult]:
        """
        批量分析多支股票
        
        Args:
            stock_list: 股票數據列表
            market_context: 市場環境
            max_concurrent: 最大併行數量
            
        Returns:
            List[AIAnalysisResult]: 分析結果列表
        """
        
        async def analyze_batch():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def analyze_single(stock_data):
                async with semaphore:
                    return await self.analyze_stock_comprehensive(stock_data, market_context)
            
            tasks = [analyze_single(stock) for stock in stock_list]
            return await asyncio.gather(*tasks)
        
        # 執行批量分析
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(analyze_batch())
            return results
        finally:
            loop.close()
    
    def generate_ranking_report(self, 
                              analysis_results: List[AIAnalysisResult],
                              top_n: int = 10) -> Dict[str, Any]:
        """
        生成股票排名報告
        
        Args:
            analysis_results: AI分析結果列表
            top_n: 返回前N名
            
        Returns:
            Dict: 排名報告
        """
        
        # 按分數排序
        sorted_results = sorted(analysis_results, 
                              key=lambda x: x.overall_score, reverse=True)
        
        top_stocks = sorted_results[:top_n]
        
        # 統計分析
        stats = {
            "total_analyzed": len(analysis_results),
            "avg_score": sum(r.overall_score for r in analysis_results) / len(analysis_results),
            "high_confidence_count": len([r for r in analysis_results if r.confidence_level > 0.7]),
            "recommended_count": len([r for r in analysis_results if r.recommendation in ['強烈推薦', '推薦']])
        }
        
        # 風險分布
        risk_distribution = {}
        for result in analysis_results:
            if result.risk_warnings:
                risk_count = len(result.risk_warnings)
                risk_level = "高風險" if risk_count >= 3 else "中風險" if risk_count >= 2 else "低風險"
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        return {
            "top_recommendations": [
                {
                    "symbol": result.stock_symbol,
                    "score": result.overall_score,
                    "recommendation": result.recommendation,
                    "key_insight": result.key_insights[0] if result.key_insights else "",
                    "confidence": result.confidence_level
                }
                for result in top_stocks
            ],
            "market_statistics": stats,
            "risk_distribution": risk_distribution,
            "analysis_summary": {
                "strongest_opportunities": [r.stock_symbol for r in top_stocks[:3]],
                "high_risk_stocks": [r.stock_symbol for r in analysis_results 
                                   if len(r.risk_warnings) >= 3],
                "high_confidence_picks": [r.stock_symbol for r in analysis_results 
                                        if r.confidence_level > 0.8]
            }
        }
    
    def get_ai_model_info(self) -> Dict[str, Any]:
        """獲取AI模型資訊"""
        
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "analysis_framework": self.analysis_framework,
            "capabilities": [
                "技術分析整合",
                "風險評估",
                "策略建議",
                "市場環境分析",
                "批量股票分析",
                "排名報告生成"
            ]
        }