"""
智能報告生成器
Intelligent Report Generator - 專業交易報告自動生成
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import json

from .stock_analyzer import AIAnalysisResult
from ..models.stock_data import ComprehensiveStockData, MarketContext
from ..risk import RiskAssessment, PositionCalculation


@dataclass
class TradingReport:
    """交易報告"""
    report_id: str
    timestamp: datetime
    market_overview: Dict[str, Any]
    stock_analysis: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    position_recommendations: List[Dict[str, Any]]
    key_insights: List[str]
    warnings: List[str]
    executive_summary: str


class ReportGenerator:
    """專業交易報告生成器"""
    
    def __init__(self):
        self.report_templates = {
            "daily_screening": "每日選股報告",
            "risk_analysis": "風險分析報告", 
            "strategy_comparison": "策略比較報告",
            "portfolio_review": "投資組合檢討",
            "market_outlook": "市場展望報告"
        }
    
    def generate_daily_screening_report(self, 
                                      ai_results: List[AIAnalysisResult],
                                      market_context: MarketContext = None) -> TradingReport:
        """
        生成每日選股報告
        
        Args:
            ai_results: AI分析結果列表
            market_context: 市場環境
            
        Returns:
            TradingReport: 完整交易報告
        """
        
        report_id = f"daily_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. 市場概覽
        market_overview = self._generate_market_overview(market_context)
        
        # 2. 股票分析摘要
        stock_analysis = self._generate_stock_analysis_summary(ai_results)
        
        # 3. 風險評估摘要
        risk_assessment = self._generate_risk_summary(ai_results)
        
        # 4. 倉位建議
        position_recommendations = self._generate_position_recommendations(ai_results)
        
        # 5. 關鍵洞察
        key_insights = self._extract_key_insights(ai_results)
        
        # 6. 風險警示
        warnings = self._extract_warnings(ai_results)
        
        # 7. 執行摘要
        executive_summary = self._generate_executive_summary(
            ai_results, market_context, key_insights, warnings
        )
        
        return TradingReport(
            report_id=report_id,
            timestamp=datetime.now(timezone.utc),
            market_overview=market_overview,
            stock_analysis=stock_analysis,
            risk_assessment=risk_assessment,
            position_recommendations=position_recommendations,
            key_insights=key_insights,
            warnings=warnings,
            executive_summary=executive_summary
        )
    
    def _generate_market_overview(self, market_context: MarketContext = None) -> Dict[str, Any]:
        """生成市場概覽"""
        
        overview = {
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_status": "開盤中" if self._is_market_open() else "休市",
        }
        
        if market_context:
            overview.update({
                "market_trend": "多頭" if market_context.is_market_bullish else "空頭",
                "volatility_environment": self._translate_volatility(market_context.volatility_environment),
                "trading_session": self._translate_session(market_context.trading_session),
                "market_sentiment": getattr(market_context, 'sentiment', '中性')
            })
        else:
            overview.update({
                "market_trend": "資料不足",
                "volatility_environment": "未知",
                "trading_session": "未知",
                "market_sentiment": "中性"
            })
        
        return overview
    
    def _generate_stock_analysis_summary(self, ai_results: List[AIAnalysisResult]) -> List[Dict[str, Any]]:
        """生成股票分析摘要"""
        
        # 按分數排序
        sorted_results = sorted(ai_results, key=lambda x: x.overall_score, reverse=True)
        
        summary = []
        for i, result in enumerate(sorted_results[:20]):  # 最多20支股票
            summary.append({
                "rank": i + 1,
                "symbol": result.stock_symbol,
                "score": f"{result.overall_score:.1f}",
                "recommendation": result.recommendation,
                "strategy": result.strategy_advice,
                "entry_timing": result.entry_timing,
                "confidence": f"{result.confidence_level:.1%}",
                "key_insight": result.key_insights[0] if result.key_insights else "無特殊洞察",
                "risk_level": "高" if len(result.risk_warnings) >= 3 else "中" if len(result.risk_warnings) >= 2 else "低"
            })
        
        return summary
    
    def _generate_risk_summary(self, ai_results: List[AIAnalysisResult]) -> Dict[str, Any]:
        """生成風險評估摘要"""
        
        total_stocks = len(ai_results)
        high_risk_count = len([r for r in ai_results if len(r.risk_warnings) >= 3])
        low_confidence_count = len([r for r in ai_results if r.confidence_level < 0.5])
        
        # 最常見的風險因子
        all_warnings = []
        for result in ai_results:
            all_warnings.extend(result.risk_warnings)
        
        warning_counts = {}
        for warning in all_warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        top_risks = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_analyzed": total_stocks,
            "high_risk_stocks": high_risk_count,
            "high_risk_percentage": f"{high_risk_count/total_stocks:.1%}" if total_stocks > 0 else "0%",
            "low_confidence_stocks": low_confidence_count,
            "average_confidence": f"{sum(r.confidence_level for r in ai_results)/total_stocks:.1%}" if total_stocks > 0 else "0%",
            "top_risk_factors": [{"risk": risk, "frequency": count} for risk, count in top_risks],
            "overall_market_risk": self._assess_overall_market_risk(ai_results)
        }
    
    def _generate_position_recommendations(self, ai_results: List[AIAnalysisResult]) -> List[Dict[str, Any]]:
        """生成倉位建議"""
        
        # 篩選推薦的股票
        recommended_stocks = [
            result for result in ai_results 
            if result.recommendation in ['強烈推薦', '推薦'] and result.confidence_level > 0.6
        ]
        
        # 按綜合評分排序
        sorted_recommendations = sorted(
            recommended_stocks, 
            key=lambda x: x.overall_score * x.confidence_level, 
            reverse=True
        )
        
        recommendations = []
        for i, result in enumerate(sorted_recommendations[:10]):  # 最多10個推薦
            recommendations.append({
                "priority": i + 1,
                "symbol": result.stock_symbol,
                "action": result.recommendation,
                "suggested_weight": self._calculate_suggested_weight(result, i),
                "entry_strategy": result.strategy_advice,
                "timing": result.entry_timing,
                "rationale": result.reasoning[:100] + "..." if len(result.reasoning) > 100 else result.reasoning
            })
        
        return recommendations
    
    def _extract_key_insights(self, ai_results: List[AIAnalysisResult]) -> List[str]:
        """提取關鍵洞察"""
        
        insights = []
        
        # 統計分析
        total_stocks = len(ai_results)
        
        # 如果沒有分析結果，返回默認洞察
        if total_stocks == 0:
            return ["暫無分析結果", "建議稍後重試或調整篩選條件"]
        
        strong_buy_count = len([r for r in ai_results if r.recommendation == '強烈推薦'])
        avg_score = sum(r.overall_score for r in ai_results) / total_stocks if total_stocks > 0 else 0
        
        insights.append(f"本次分析{total_stocks}支股票，平均適合度{avg_score:.1f}分")
        
        if strong_buy_count > 0:
            insights.append(f"發現{strong_buy_count}支強烈推薦標的")
        
        # 提取最常見的洞察
        all_insights = []
        for result in ai_results:
            all_insights.extend(result.key_insights)
        
        insight_counts = {}
        for insight in all_insights:
            if len(insight) > 10:  # 篩選有意義的洞察
                insight_counts[insight] = insight_counts.get(insight, 0) + 1
        
        top_insights = sorted(insight_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for insight, count in top_insights:
            if count >= 2:  # 至少出現2次
                insights.append(f"共同趨勢: {insight}")
        
        # 市場狀況洞察
        high_score_count = len([r for r in ai_results if r.overall_score > 75])
        if total_stocks > 0:  # 防止除零錯誤
            if high_score_count / total_stocks > 0.3:
                insights.append("市場機會豐富，多數標的表現良好")
            elif high_score_count / total_stocks < 0.1:
                insights.append("市場機會有限，建議謹慎操作")
        
        return insights[:8]  # 最多8個洞察
    
    def _extract_warnings(self, ai_results: List[AIAnalysisResult]) -> List[str]:
        """提取風險警示"""
        
        warnings = []
        
        # 統計風險分布
        total_stocks = len(ai_results)
        
        # 如果沒有分析結果，返回默認警示
        if total_stocks == 0:
            return ["暫無分析數據", "請先執行股票分析"]
        
        high_risk_count = len([r for r in ai_results if len(r.risk_warnings) >= 3])
        low_confidence_count = len([r for r in ai_results if r.confidence_level < 0.5])
        
        if high_risk_count / total_stocks > 0.4:
            warnings.append("超過40%的標的存在高風險，建議降低整體曝險")
        
        if low_confidence_count / total_stocks > 0.3:
            warnings.append("多數分析信心度偏低，市場不確定性較高")
        
        # 提取最常見的警示
        all_warnings = []
        for result in ai_results:
            all_warnings.extend(result.risk_warnings)
        
        warning_counts = {}
        for warning in all_warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        top_warnings = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for warning, count in top_warnings:
            if count >= 3:  # 至少出現3次
                warnings.append(f"普遍風險: {warning}")
        
        # 特殊警示
        avoid_count = len([r for r in ai_results if r.recommendation == '避免'])
        if avoid_count > 0:
            warnings.append(f"有{avoid_count}支股票建議避免交易")
        
        return warnings[:6]  # 最多6個警示
    
    def _generate_executive_summary(self, 
                                  ai_results: List[AIAnalysisResult],
                                  market_context: MarketContext = None,
                                  key_insights: List[str] = None,
                                  warnings: List[str] = None) -> str:
        """生成執行摘要"""
        
        total_stocks = len(ai_results)
        recommended_count = len([r for r in ai_results if r.recommendation in ['強烈推薦', '推薦']])
        avg_score = sum(r.overall_score for r in ai_results) / total_stocks if total_stocks > 0 else 0
        avg_confidence = sum(r.confidence_level for r in ai_results) / total_stocks if total_stocks > 0 else 0
        
        summary_parts = []
        
        # 基本統計
        summary_parts.append(
            f"本次分析涵蓋{total_stocks}支台股，平均適合度{avg_score:.1f}分，"
            f"整體分析信心度{avg_confidence:.1%}。"
        )
        
        # 推薦情況
        if recommended_count > 0:
            summary_parts.append(
                f"共發現{recommended_count}支值得關注的標的，"
                f"佔總數{recommended_count/total_stocks:.1%}。"
            )
        else:
            summary_parts.append("當前市況下，暫無明確推薦標的。")
        
        # 市場環境
        if market_context:
            trend_desc = "多頭" if market_context.is_market_bullish else "空頭"
            summary_parts.append(f"大盤呈現{trend_desc}格局。")
        
        # 關鍵洞察
        if key_insights:
            top_insight = key_insights[0] if key_insights else ""
            if "共同趨勢" in top_insight:
                summary_parts.append(top_insight)
        
        # 風險提醒
        if warnings:
            if any("高風險" in w for w in warnings):
                summary_parts.append("注意當前市場風險偏高，建議謹慎操作。")
            elif any("不確定性" in w for w in warnings):
                summary_parts.append("市場不確定性較高，建議分散投資。")
        
        # 操作建議
        if avg_score > 70:
            summary_parts.append("整體機會良好，可積極布局。")
        elif avg_score > 50:
            summary_parts.append("市場機會適中，建議精選個股。")
        else:
            summary_parts.append("市場機會有限，建議保守觀望。")
        
        return " ".join(summary_parts)
    
    def _is_market_open(self) -> bool:
        """檢查市場是否開盤"""
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute
        
        # 週一到週五
        if 0 <= weekday <= 4:
            # 09:00-13:30
            if (hour == 9 and minute >= 0) or (9 < hour < 13) or (hour == 13 and minute <= 30):
                return True
        
        return False
    
    def _translate_volatility(self, volatility: str) -> str:
        """翻譯波動性描述"""
        translations = {
            "low_volatility": "低波動",
            "medium_volatility": "中波動", 
            "high_volatility": "高波動"
        }
        return translations.get(volatility, volatility)
    
    def _translate_session(self, session: str) -> str:
        """翻譯交易時段"""
        translations = {
            "pre_market": "盤前",
            "opening": "開盤",
            "trading": "盤中",
            "closing": "收盤",
            "after_hours": "盤後"
        }
        return translations.get(session, session)
    
    def _assess_overall_market_risk(self, ai_results: List[AIAnalysisResult]) -> str:
        """評估整體市場風險"""
        
        total_stocks = len(ai_results)
        if total_stocks == 0:
            return "無法評估"
        
        high_risk_ratio = len([r for r in ai_results if len(r.risk_warnings) >= 3]) / total_stocks
        low_confidence_ratio = len([r for r in ai_results if r.confidence_level < 0.5]) / total_stocks
        
        if high_risk_ratio > 0.5 or low_confidence_ratio > 0.5:
            return "高風險"
        elif high_risk_ratio > 0.3 or low_confidence_ratio > 0.3:
            return "中高風險"
        elif high_risk_ratio > 0.2 or low_confidence_ratio > 0.2:
            return "中等風險"
        else:
            return "低風險"
    
    def _calculate_suggested_weight(self, result: AIAnalysisResult, rank: int) -> str:
        """計算建議權重"""
        
        base_weight = max(2, 8 - rank)  # 排名越前權重越高
        confidence_multiplier = result.confidence_level
        score_multiplier = result.overall_score / 100
        
        final_weight = base_weight * confidence_multiplier * score_multiplier
        final_weight = min(15, max(2, final_weight))  # 限制在2%-15%
        
        return f"{final_weight:.1f}%"
    
    def export_to_dict(self, report: TradingReport) -> Dict[str, Any]:
        """導出報告為字典格式"""
        
        return {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "market_overview": report.market_overview,
            "stock_analysis": report.stock_analysis,
            "risk_assessment": report.risk_assessment,
            "position_recommendations": report.position_recommendations,
            "key_insights": report.key_insights,
            "warnings": report.warnings,
            "executive_summary": report.executive_summary
        }
    
    def export_to_json(self, report: TradingReport, file_path: str = None) -> str:
        """導出報告為JSON格式"""
        
        report_dict = self.export_to_dict(report)
        json_str = json.dumps(report_dict, ensure_ascii=False, indent=2)
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str
    
    def generate_markdown_report(self, report: TradingReport) -> str:
        """生成Markdown格式報告"""
        
        md_content = f"""# 台股當沖選股分析報告
        
## 報告摘要
**報告編號**: {report.report_id}  
**生成時間**: {report.timestamp.strftime("%Y-%m-%d %H:%M:%S")}  

{report.executive_summary}

## 市場概覽
"""
        
        for key, value in report.market_overview.items():
            md_content += f"- **{key}**: {value}\n"
        
        md_content += "\n## 重點推薦標的\n\n"
        md_content += "| 排名 | 股票代號 | 評分 | 建議 | 信心度 | 關鍵洞察 |\n"
        md_content += "|------|----------|------|------|--------|----------|\n"
        
        for stock in report.stock_analysis[:10]:
            md_content += f"| {stock['rank']} | {stock['symbol']} | {stock['score']} | {stock['recommendation']} | {stock['confidence']} | {stock['key_insight'][:30]}... |\n"
        
        md_content += "\n## 風險評估\n\n"
        md_content += f"- **總分析標的**: {report.risk_assessment['total_analyzed']}支\n"
        md_content += f"- **高風險標的**: {report.risk_assessment['high_risk_stocks']}支 ({report.risk_assessment['high_risk_percentage']})\n"
        md_content += f"- **平均信心度**: {report.risk_assessment['average_confidence']}\n"
        md_content += f"- **整體市場風險**: {report.risk_assessment['overall_market_risk']}\n"
        
        md_content += "\n## 關鍵洞察\n\n"
        for insight in report.key_insights:
            md_content += f"- {insight}\n"
        
        md_content += "\n## 風險警示\n\n"
        for warning in report.warnings:
            md_content += f"⚠️ {warning}\n"
        
        md_content += "\n## 倉位建議\n\n"
        for rec in report.position_recommendations:
            md_content += f"### {rec['symbol']} - {rec['action']}\n"
            md_content += f"- **建議權重**: {rec['suggested_weight']}\n"
            md_content += f"- **進場策略**: {rec['entry_strategy']}\n"
            md_content += f"- **時機建議**: {rec['timing']}\n\n"
        
        return md_content