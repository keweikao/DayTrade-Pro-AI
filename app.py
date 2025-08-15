"""
台股當沖AI選股系統
Taiwan Stock Day Trading AI Screening System
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import asyncio
import json
from typing import List, Dict, Any
import os

# 導入自定義模組
from src.data.taiwan_market_data import TaiwanMarketDataProvider
from src.ai.stock_analyzer import AIStockAnalyzer
from src.ai.report_generator import ReportGenerator
from src.risk.fund_allocator import FundAllocationCalculator, PositionSizeMethod
from src.models.stock_data import MarketContext

# 設定頁面配置
st.set_page_config(
    page_title="DayTrade Pro AI - 專業當沖智能分析",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TaiwanStockApp:
    """台股分析應用程式主類別"""
    
    def __init__(self):
        self.data_provider = TaiwanMarketDataProvider()
        self.report_generator = ReportGenerator()
        self.fund_allocator = FundAllocationCalculator()
        
        # 初始化session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        if 'current_report' not in st.session_state:
            st.session_state.current_report = None
        if 'openai_api_key' not in st.session_state:
            # 優先從環境變數讀取API Key
            st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY', "")
    
    def run(self):
        """運行主應用程式"""
        
        # 主標題
        st.markdown('<h1 class="main-header">⚡ DayTrade Pro AI</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-top: -1rem;">專業當沖智能分析系統 | 精準交易，智勝市場</p>', 
                   unsafe_allow_html=True)
        
        # 側邊欄設定
        self.setup_sidebar()
        
        # 主要內容區域
        if st.session_state.openai_api_key:
            self.main_content()
        else:
            self.show_api_key_required()
    
    def setup_sidebar(self):
        """設定側邊欄"""
        
        st.sidebar.header("⚙️ 系統設定")
        
        # API Key設定
        st.sidebar.subheader("OpenAI API 設定")
        
        # 檢查是否有環境變數中的API Key
        env_api_key = os.getenv('OPENAI_API_KEY', "")
        if env_api_key:
            st.sidebar.success("✅ 使用環境變數中的API Key")
            st.session_state.openai_api_key = env_api_key
        else:
            api_key = st.sidebar.text_input(
                "請輸入OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                help="需要OpenAI API Key才能使用AI分析功能"
            )
            
            if api_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = api_key
                if api_key:
                    st.sidebar.success("✅ API Key已設定")
        
        # 資金設定
        st.sidebar.subheader("💰 資金管理")
        total_capital = st.sidebar.number_input(
            "總資金 (萬元)",
            min_value=10,
            max_value=10000,
            value=100,
            step=10
        ) * 10000
        
        self.fund_allocator.total_capital = total_capital
        self.fund_allocator.available_capital = total_capital
        
        # 風險設定
        st.sidebar.subheader("⚠️ 風險控制")
        max_risk_per_trade = st.sidebar.slider(
            "單筆最大風險 (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        ) / 100
        
        max_single_position = st.sidebar.slider(
            "單一持股上限 (%)",
            min_value=5.0,
            max_value=25.0,
            value=15.0,
            step=1.0
        ) / 100
        
        self.fund_allocator.max_risk_per_trade = max_risk_per_trade
        self.fund_allocator.max_single_position = max_single_position
        
        # 顯示設定摘要
        st.sidebar.markdown("---")
        st.sidebar.markdown("📊 **當前設定摘要**")
        st.sidebar.write(f"💰 總資金: {total_capital:,.0f} 元")
        st.sidebar.write(f"⚠️ 單筆風險: {max_risk_per_trade:.1%}")
        st.sidebar.write(f"📈 持股上限: {max_single_position:.1%}")
    
    def show_api_key_required(self):
        """顯示需要API Key的提示"""
        
        st.markdown("""
        ## 🔑 需要設定API Key
        
        本系統使用OpenAI GPT-4進行智能股票分析，請在左側邊欄輸入您的OpenAI API Key。
        
        ### 如何獲取API Key：
        1. 訪問 [OpenAI官網](https://platform.openai.com/)
        2. 註冊或登錄您的帳戶
        3. 前往API Keys頁面
        4. 創建新的API Key
        5. 將Key複製到左側邊欄
        
        ### 注意事項：
        - API Key會保存在當前session中，不會永久儲存
        - 使用AI分析功能會產生API調用費用
        - 建議設定適當的使用限額
        """)
        
        # 顯示系統功能預覽
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🎯 智能選股
            - 多維度技術分析
            - AI驅動的策略識別
            - 即時風險評估
            """)
        
        with col2:
            st.markdown("""
            #### 📊 專業報告
            - 詳細分析報告
            - 風險警示系統
            - 倉位管理建議
            """)
        
        with col3:
            st.markdown("""
            #### ⚡ 當沖專門
            - 台股交易特性
            - 實時市場分析
            - 專業級風控
            """)
    
    def main_content(self):
        """主要內容區域"""
        
        # 頁籤選擇
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 股票分析", "📊 分析報告", "💰 倉位管理", "📈 系統監控"
        ])
        
        with tab1:
            self.stock_analysis_tab()
        
        with tab2:
            self.analysis_report_tab()
        
        with tab3:
            self.position_management_tab()
        
        with tab4:
            self.system_monitor_tab()
    
    def stock_analysis_tab(self):
        """股票分析頁籤"""
        
        st.header("🎯 智能股票分析")
        
        # 股票代號輸入
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            stock_symbols = st.text_input(
                "請輸入股票代號 (多個代號用逗號分隔)",
                placeholder="例如: 2330, 2454, 2317",
                help="台股代號，例如台積電為2330"
            )
        
        with col2:
            analysis_depth = st.selectbox(
                "分析深度",
                ["標準分析", "深度分析", "快速掃描"],
                index=0
            )
        
        with col3:
            max_stocks = st.number_input(
                "最大分析數量",
                min_value=1,
                max_value=50,
                value=10
            )
        
        # 分析按鈕
        if st.button("🚀 開始AI分析", type="primary"):
            if not stock_symbols.strip():
                st.error("請輸入至少一個股票代號")
                return
            
            # 解析股票代號
            symbols = [s.strip() for s in stock_symbols.split(",") if s.strip()]
            symbols = symbols[:max_stocks]  # 限制數量
            
            with st.spinner("🤖 AI正在分析中，請稍候..."):
                self.perform_stock_analysis(symbols, analysis_depth)
        
        # 顯示分析結果
        if st.session_state.analysis_results:
            self.display_analysis_results()
    
    def perform_stock_analysis(self, symbols: List[str], depth: str):
        """執行股票分析"""
        
        try:
            # 初始化AI分析器
            ai_analyzer = AIStockAnalyzer(st.session_state.openai_api_key)
            
            # 獲取股票數據
            stock_data_list = []
            progress_bar = st.progress(0)
            
            for i, symbol in enumerate(symbols):
                try:
                    # 更新進度
                    progress_bar.progress((i + 1) / len(symbols))
                    
                    # 獲取股票數據
                    stock_data = self.data_provider.get_comprehensive_stock_data(symbol)
                    if stock_data:
                        stock_data_list.append(stock_data)
                    
                except Exception as e:
                    st.warning(f"無法獲取 {symbol} 的數據: {str(e)}")
            
            if not stock_data_list:
                st.error("無法獲取任何股票數據，請檢查代號是否正確")
                return
            
            # 執行AI分析
            st.info(f"📊 成功獲取 {len(stock_data_list)} 支股票數據，開始AI分析...")
            
            market_context = self._create_market_context()
            
            # 批量分析
            analysis_results = ai_analyzer.analyze_multiple_stocks(
                stock_data_list, 
                market_context,
                max_concurrent=3
            )
            
            # 保存結果
            st.session_state.analysis_results = analysis_results
            
            # 生成報告
            report = self.report_generator.generate_daily_screening_report(
                analysis_results, market_context
            )
            st.session_state.current_report = report
            
            st.success(f"✅ 分析完成！共分析 {len(analysis_results)} 支股票")
            
        except Exception as e:
            st.error(f"分析過程中發生錯誤: {str(e)}")
    
    def display_analysis_results(self):
        """顯示分析結果"""
        
        st.subheader("📈 分析結果")
        
        # 結果統計
        results = st.session_state.analysis_results
        total_stocks = len(results)
        recommended_stocks = len([r for r in results if r.recommendation in ['強烈推薦', '推薦']])
        avg_score = sum(r.overall_score for r in results) / total_stocks if total_stocks > 0 else 0
        
        # 顯示關鍵指標
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("總分析股票", total_stocks)
        
        with col2:
            st.metric("推薦標的", recommended_stocks)
        
        with col3:
            st.metric("平均評分", f"{avg_score:.1f}")
        
        with col4:
            confidence_avg = sum(r.confidence_level for r in results) / total_stocks if total_stocks > 0 else 0
            st.metric("平均信心度", f"{confidence_avg:.1%}")
        
        # 排序選項
        sort_option = st.selectbox(
            "排序方式",
            ["綜合評分", "推薦等級", "信心度", "股票代號"],
            index=0
        )
        
        # 排序結果
        if sort_option == "綜合評分":
            sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
        elif sort_option == "推薦等級":
            recommendation_order = {'強烈推薦': 5, '推薦': 4, '中性': 3, '謹慎': 2, '避免': 1}
            sorted_results = sorted(results, 
                                  key=lambda x: recommendation_order.get(x.recommendation, 0), 
                                  reverse=True)
        elif sort_option == "信心度":
            sorted_results = sorted(results, key=lambda x: x.confidence_level, reverse=True)
        else:
            sorted_results = sorted(results, key=lambda x: x.stock_symbol)
        
        # 結果表格
        st.subheader("📋 詳細分析結果")
        
        for i, result in enumerate(sorted_results):
            with st.expander(f"#{i+1} {result.stock_symbol} - {result.recommendation} ({result.overall_score:.1f}分)"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**基本資訊**")
                    st.write(f"📊 綜合評分: {result.overall_score:.1f}/100")
                    st.write(f"🎯 投資建議: {result.recommendation}")
                    st.write(f"🔮 信心度: {result.confidence_level:.1%}")
                    st.write(f"⏰ 進場時機: {result.entry_timing}")
                
                with col2:
                    st.markdown("**策略建議**")
                    st.write(result.strategy_advice)
                
                # 關鍵洞察
                if result.key_insights:
                    st.markdown("**💡 關鍵洞察**")
                    for insight in result.key_insights:
                        st.write(f"• {insight}")
                
                # 風險警示
                if result.risk_warnings:
                    st.markdown("**⚠️ 風險警示**")
                    for warning in result.risk_warnings:
                        st.write(f"⚠️ {warning}")
                
                # 分析邏輯
                if result.reasoning:
                    st.markdown("**🧠 分析邏輯**")
                    st.write(result.reasoning)
    
    def analysis_report_tab(self):
        """分析報告頁籤"""
        
        st.header("📊 專業分析報告")
        
        if not st.session_state.current_report:
            st.info("請先在「股票分析」頁籤執行分析，系統會自動生成報告")
            return
        
        report = st.session_state.current_report
        
        # 報告摘要
        st.subheader("📋 執行摘要")
        st.markdown(f'<div class="success-box">{report.executive_summary}</div>', 
                   unsafe_allow_html=True)
        
        # 市場概覽
        st.subheader("🌐 市場概覽")
        col1, col2 = st.columns(2)
        
        with col1:
            for key, value in list(report.market_overview.items())[:len(report.market_overview)//2]:
                st.write(f"**{key}**: {value}")
        
        with col2:
            for key, value in list(report.market_overview.items())[len(report.market_overview)//2:]:
                st.write(f"**{key}**: {value}")
        
        # 關鍵洞察
        if report.key_insights:
            st.subheader("💡 關鍵洞察")
            for insight in report.key_insights:
                st.write(f"✨ {insight}")
        
        # 風險警示
        if report.warnings:
            st.subheader("⚠️ 風險警示")
            for warning in report.warnings:
                st.markdown(f'<div class="warning-box">⚠️ {warning}</div>', 
                           unsafe_allow_html=True)
        
        # 風險評估詳情
        st.subheader("🛡️ 風險評估")
        risk = report.risk_assessment
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("總分析標的", risk['total_analyzed'])
        with col2:
            st.metric("高風險標的", risk['high_risk_stocks'])
        with col3:
            st.metric("整體市場風險", risk['overall_market_risk'])
        
        # 倉位建議
        if report.position_recommendations:
            st.subheader("💰 倉位建議")
            
            # 建議表格
            recommendations_df = pd.DataFrame([
                {
                    "優先級": rec["priority"],
                    "股票代號": rec["symbol"],
                    "操作建議": rec["action"],
                    "建議權重": rec["suggested_weight"],
                    "進場策略": rec["entry_strategy"][:50] + "..." if len(rec["entry_strategy"]) > 50 else rec["entry_strategy"],
                    "時機建議": rec["timing"]
                }
                for rec in report.position_recommendations
            ])
            
            st.dataframe(recommendations_df, use_container_width=True)
        
        # 報告導出
        st.subheader("📤 報告導出")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 導出JSON"):
                json_str = self.report_generator.export_to_json(report)
                st.download_button(
                    label="下載JSON報告",
                    data=json_str,
                    file_name=f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📝 導出Markdown"):
                md_content = self.report_generator.generate_markdown_report(report)
                st.download_button(
                    label="下載Markdown報告",
                    data=md_content,
                    file_name=f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col3:
            if st.button("📊 顯示圖表"):
                self.show_analysis_charts()
    
    def position_management_tab(self):
        """倉位管理頁籤"""
        
        st.header("💰 智能倉位管理")
        
        if not st.session_state.analysis_results:
            st.info("請先執行股票分析")
            return
        
        # 篩選推薦股票
        results = st.session_state.analysis_results
        recommended_stocks = [
            r for r in results 
            if r.recommendation in ['強烈推薦', '推薦'] and r.confidence_level > 0.5
        ]
        
        if not recommended_stocks:
            st.warning("目前沒有符合條件的推薦股票")
            return
        
        st.subheader("🎯 推薦標的倉位計算")
        
        # 倉位計算方法選擇
        position_method = st.selectbox(
            "倉位計算方法",
            [method.value for method in PositionSizeMethod],
            index=1  # 默認ATR方法
        )
        
        # 最大持股數量
        max_positions = st.slider(
            "最大持股數量",
            min_value=1,
            max_value=min(10, len(recommended_stocks)),
            value=min(5, len(recommended_stocks))
        )
        
        # 模擬倉位計算
        if st.button("🔢 計算建議倉位", type="primary"):
            self.calculate_position_allocations(recommended_stocks, position_method, max_positions)
        
        # 顯示倉位建議
        if 'position_allocations' in st.session_state:
            self.display_position_allocations()
    
    def calculate_position_allocations(self, recommended_stocks, method_name, max_positions):
        """計算倉位配置"""
        
        try:
            # 轉換方法枚舉
            method = next(m for m in PositionSizeMethod if m.value == method_name)
            
            # 準備候選股票數據（簡化版）
            candidates = []
            for result in recommended_stocks[:max_positions]:
                # 注意：實際應用中需要完整的股票數據和風險評估
                candidates.append({
                    'symbol': result.stock_symbol,
                    'score': result.overall_score,
                    'stock_data': None,  # 簡化版本
                    'risk_assessment': None,  # 簡化版本
                    'strategy_info': {
                        'strategy_name': '綜合策略',
                        'is_intraday': True,
                        'expected_win_rate': result.confidence_level
                    }
                })
            
            # 簡化的倉位計算
            allocations = []
            total_capital = self.fund_allocator.total_capital
            
            for i, candidate in enumerate(candidates):
                # 基於排名和分數的簡化計算
                base_weight = max(0.02, 0.15 - (i * 0.02))  # 2%-15%
                score_multiplier = candidate['score'] / 100
                final_weight = base_weight * score_multiplier
                
                position_value = total_capital * final_weight
                
                allocations.append({
                    'symbol': candidate['symbol'],
                    'rank': i + 1,
                    'score': candidate['score'],
                    'weight': final_weight,
                    'position_value': position_value,
                    'method': method_name
                })
            
            st.session_state.position_allocations = allocations
            
        except Exception as e:
            st.error(f"倉位計算錯誤: {str(e)}")
    
    def display_position_allocations(self):
        """顯示倉位配置"""
        
        allocations = st.session_state.position_allocations
        
        st.subheader("📊 倉位配置結果")
        
        # 總覽
        total_allocation = sum(alloc['position_value'] for alloc in allocations)
        cash_reserved = self.fund_allocator.total_capital - total_allocation
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("總配置金額", f"{total_allocation:,.0f} 元")
        with col2:
            st.metric("保留現金", f"{cash_reserved:,.0f} 元")
        with col3:
            st.metric("資金利用率", f"{total_allocation/self.fund_allocator.total_capital:.1%}")
        
        # 配置表格
        allocation_df = pd.DataFrame([
            {
                "排名": alloc["rank"],
                "股票代號": alloc["symbol"],
                "評分": f"{alloc['score']:.1f}",
                "建議權重": f"{alloc['weight']:.1%}",
                "投入金額": f"{alloc['position_value']:,.0f}",
                "計算方法": alloc["method"]
            }
            for alloc in allocations
        ])
        
        st.dataframe(allocation_df, use_container_width=True)
        
        # 權重分布圖
        fig = px.pie(
            values=[alloc['weight'] for alloc in allocations],
            names=[alloc['symbol'] for alloc in allocations],
            title="倉位權重分布"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def system_monitor_tab(self):
        """系統監控頁籤"""
        
        st.header("📈 系統監控")
        
        # 系統狀態
        st.subheader("🔧 系統狀態")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📡 數據連接**")
            try:
                # 測試數據連接
                test_data = self.data_provider.get_comprehensive_stock_data("2330")
                if test_data:
                    st.success("✅ 數據源正常")
                else:
                    st.warning("⚠️ 數據源異常")
            except:
                st.error("❌ 數據源錯誤")
        
        with col2:
            st.markdown("**🤖 AI服務**")
            if st.session_state.openai_api_key:
                st.success("✅ API Key已設定")
            else:
                st.error("❌ 缺少API Key")
        
        with col3:
            st.markdown("**💾 Session狀態**")
            if st.session_state.analysis_results:
                st.success(f"✅ 已分析 {len(st.session_state.analysis_results)} 支股票")
            else:
                st.info("ℹ️ 尚未執行分析")
        
        # 性能指標
        st.subheader("📊 使用統計")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("分析股票數", len(st.session_state.analysis_results))
        
        with col2:
            if st.session_state.analysis_results:
                avg_score = sum(r.overall_score for r in st.session_state.analysis_results) / len(st.session_state.analysis_results)
                st.metric("平均評分", f"{avg_score:.1f}")
            else:
                st.metric("平均評分", "N/A")
        
        with col3:
            st.metric("總資金", f"{self.fund_allocator.total_capital:,.0f}")
        
        with col4:
            if st.session_state.current_report:
                st.metric("最新報告", "可用")
            else:
                st.metric("最新報告", "無")
        
        # 系統設定摘要
        st.subheader("⚙️ 當前設定")
        
        settings_data = {
            "總資金": f"{self.fund_allocator.total_capital:,.0f} 元",
            "單筆最大風險": f"{self.fund_allocator.max_risk_per_trade:.1%}",
            "單一持股上限": f"{self.fund_allocator.max_single_position:.1%}",
            "保留現金比例": f"{self.fund_allocator.reserved_cash_ratio:.1%}",
            "當沖槓桿": f"{self.fund_allocator.intraday_leverage:.1f}x",
        }
        
        settings_df = pd.DataFrame([
            {"設定項目": key, "當前值": value}
            for key, value in settings_data.items()
        ])
        
        st.dataframe(settings_df, use_container_width=True)
        
        # 重設功能
        st.subheader("🔄 系統重設")
        if st.button("清除所有分析結果", type="secondary"):
            st.session_state.analysis_results = []
            st.session_state.current_report = None
            if 'position_allocations' in st.session_state:
                del st.session_state.position_allocations
            st.success("✅ 已清除所有分析結果")
            st.experimental_rerun()
    
    def show_analysis_charts(self):
        """顯示分析圖表"""
        
        if not st.session_state.analysis_results:
            return
        
        results = st.session_state.analysis_results
        
        # 評分分布圖
        scores = [r.overall_score for r in results]
        fig1 = px.histogram(
            x=scores,
            nbins=20,
            title="股票評分分布",
            labels={'x': '評分', 'y': '股票數量'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # 推薦分布圖
        recommendations = [r.recommendation for r in results]
        rec_counts = pd.Series(recommendations).value_counts()
        
        fig2 = px.bar(
            x=rec_counts.index,
            y=rec_counts.values,
            title="推薦等級分布",
            labels={'x': '推薦等級', 'y': '股票數量'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # 信心度vs評分散點圖
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=[r.confidence_level for r in results],
            y=[r.overall_score for r in results],
            mode='markers',
            text=[r.stock_symbol for r in results],
            textposition="top center",
            marker=dict(
                size=10,
                color=[len(r.risk_warnings) for r in results],
                colorscale='RdYlGn_r',
                colorbar=dict(title="風險警示數量")
            )
        ))
        
        fig3.update_layout(
            title="信心度 vs 評分關係圖",
            xaxis_title="信心度",
            yaxis_title="評分"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    def _create_market_context(self) -> MarketContext:
        """創建市場環境上下文"""
        
        # 簡化的市場環境（實際應用中應從真實數據獲取）
        return MarketContext(
            taiex=17000.0,  # 假設台指點位
            taiex_change=50.0,  # 假設上漲50點
            taiex_change_pct=0.003,  # 上漲0.3%
            vix=18.5,  # VIX指數
            foreign_investment=15.2,  # 外資買超15.2億
            sector_performance={
                "半導體": 0.012,
                "金融": -0.005,
                "電子": 0.008
            },
            market_sentiment="neutral",
            trading_session="trading"
        )

# 主程式入口
if __name__ == "__main__":
    app = TaiwanStockApp()
    app.run()