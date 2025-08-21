"""
Interactive Streamlit Dashboard for Ethiopian NER System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from vendor_analytics.scorecard import VendorScorecard
from interpretability.model_explainer import NERModelExplainer


def load_data():
    """Load available data files"""
    data_files = {}
    
    # Check for processed data
    processed_path = "data/processed/"
    if os.path.exists(processed_path):
        for file in os.listdir(processed_path):
            if file.endswith('.csv'):
                data_files[f"Processed: {file}"] = os.path.join(processed_path, file)
    
    # Check for raw data
    raw_path = "data/raw/"
    if os.path.exists(raw_path):
        for file in os.listdir(raw_path):
            if file.endswith('.csv'):
                data_files[f"Raw: {file}"] = os.path.join(raw_path, file)
    
    return data_files


def load_model_results():
    """Load model comparison results"""
    results_path = "models/comparison_report.json"
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_vendor_analytics():
    """Load vendor analytics results"""
    analytics_path = "models/vendor_analytics/lending_report.json"
    if os.path.exists(analytics_path):
        with open(analytics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Ethiopian Telegram NER System",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🛍️ Ethiopian E-commerce NER System")
    st.markdown("**Comprehensive Named Entity Recognition and Vendor Analytics for Ethiopian Telegram Channels**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Data Explorer", "Model Performance", "Vendor Analytics", "NER Playground", "Interpretability"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Vendor Analytics":
        show_vendor_analytics()
    elif page == "NER Playground":
        show_ner_playground()
    elif page == "Interpretability":
        show_interpretability()


def show_overview():
    """Show system overview"""
    st.header("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Load basic stats
    data_files = load_data()
    model_results = load_model_results()
    vendor_results = load_vendor_analytics()
    
    with col1:
        st.metric("Data Files", len(data_files))
    
    with col2:
        models_count = len(model_results.get('results', {})) if model_results else 0
        st.metric("Models Trained", models_count)
    
    with col3:
        vendors_count = vendor_results.get('summary', {}).get('total_vendors_analyzed', 0) if vendor_results else 0
        st.metric("Vendors Analyzed", vendors_count)
    
    with col4:
        high_priority = 0
        if vendor_results:
            risk_dist = vendor_results.get('summary', {}).get('risk_distribution', {})
            high_priority = risk_dist.get('High Priority', 0)
        st.metric("High Priority Vendors", high_priority)
    
    # System Architecture
    st.subheader("🏗️ System Architecture")
    
    architecture_flow = """
    ```
    📱 Telegram Channels → 🔄 Data Ingestion → 📝 Text Preprocessing
                                                        ↓
    📊 Vendor Analytics ← 🤖 NER Model Training ← 🏷️ CoNLL Labeling
                                                        ↓
    📈 Dashboard ← 🔍 Model Interpretability ← ⚖️ Model Comparison
    ```
    """
    st.markdown(architecture_flow)
    
    # Recent Activity
    st.subheader("📈 Recent Activity")
    
    if model_results:
        st.success(f"✅ Last model comparison: {model_results.get('comparison_date', 'N/A')}")
    
    if vendor_results:
        st.success(f"✅ Last vendor analysis: {vendor_results.get('summary', {}).get('report_date', 'N/A')}")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data", help="Reload all data and results"):
            st.rerun()
    
    with col2:
        st.markdown("[📖 Documentation](README.md)")
    
    with col3:
        st.markdown("[🐛 Report Issues](https://github.com/your-repo/issues)")


def show_data_explorer():
    """Show data exploration interface"""
    st.header("🔍 Data Explorer")
    
    data_files = load_data()
    
    if not data_files:
        st.warning("No data files found. Please run the data ingestion pipeline first.")
        return
    
    # File selection
    selected_file = st.selectbox("Select data file:", list(data_files.keys()))
    
    if selected_file:
        file_path = data_files[selected_file]
        
        try:
            df = pd.read_csv(file_path)
            
            # Basic statistics
            st.subheader("📊 Dataset Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Messages", len(df))
            
            with col2:
                unique_channels = df['channel'].nunique() if 'channel' in df.columns else 0
                st.metric("Unique Channels", unique_channels)
            
            with col3:
                avg_length = df['text'].str.len().mean() if 'text' in df.columns else 0
                st.metric("Avg Message Length", f"{avg_length:.0f}")
            
            with col4:
                if 'views' in df.columns:
                    avg_views = pd.to_numeric(df['views'], errors='coerce').mean()
                    st.metric("Avg Views", f"{avg_views:.0f}")
            
            # Channel distribution
            if 'channel' in df.columns:
                st.subheader("📈 Channel Distribution")
                channel_counts = df['channel'].value_counts()
                
                fig = px.bar(
                    x=channel_counts.index,
                    y=channel_counts.values,
                    title="Messages per Channel"
                )
                fig.update_xaxes(title="Channel")
                fig.update_yaxes(title="Message Count")
                st.plotly_chart(fig, use_container_width=True)
            
            # Sample data
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(10))
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"exported_{selected_file.split(':')[-1]}",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")


def show_model_performance():
    """Show model performance metrics"""
    st.header("🤖 Model Performance")
    
    model_results = load_model_results()
    
    if not model_results:
        st.warning("No model results found. Please run the model training pipeline first.")
        return
    
    # Model comparison
    st.subheader("📊 Model Comparison")
    
    results = model_results.get('results', {})
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful_results:
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in successful_results.items():
            comparison_data.append({
                'Model': model_name.split('/')[-1],
                'F1 Score': result['f1_score'],
                'Precision': result['precision'],
                'Recall': result['recall']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Metrics visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='F1 Score',
            x=comparison_df['Model'],
            y=comparison_df['F1 Score'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Precision',
            x=comparison_df['Model'],
            y=comparison_df['Precision'],
            marker_color='lightgreen'
        ))
        
        fig.add_trace(go.Bar(
            name='Recall',
            x=comparison_df['Model'],
            y=comparison_df['Recall'],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = model_results.get('summary', {}).get('best_model', 'N/A')
        if best_model != 'N/A':
            st.success(f"🏆 Best performing model: **{best_model}**")
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        st.dataframe(comparison_df)
    
    else:
        st.error("No successful model training results found.")


def show_vendor_analytics():
    """Show vendor analytics dashboard"""
    st.header("💼 Vendor Analytics & Scorecard")
    
    vendor_results = load_vendor_analytics()
    
    if not vendor_results:
        st.warning("No vendor analytics found. Please run the vendor analytics pipeline first.")
        return
    
    # Summary metrics
    summary = vendor_results.get('summary', {})
    
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vendors", summary.get('total_vendors_analyzed', 0))
    
    with col2:
        avg_score = summary.get('lending_score_stats', {}).get('mean', 0)
        st.metric("Avg Lending Score", f"{avg_score:.1f}")
    
    with col3:
        risk_dist = summary.get('risk_distribution', {})
        high_priority = risk_dist.get('High Priority', 0)
        st.metric("High Priority", high_priority)
    
    with col4:
        medium_priority = risk_dist.get('Medium Priority', 0)
        st.metric("Medium Priority", medium_priority)
    
    # Risk distribution pie chart
    st.subheader("🎯 Risk Category Distribution")
    
    risk_dist = summary.get('risk_distribution', {})
    if risk_dist:
        fig = px.pie(
            values=list(risk_dist.values()),
            names=list(risk_dist.keys()),
            title="Vendor Risk Categories",
            color_discrete_map={
                'High Priority': 'green',
                'Medium Priority': 'orange',
                'Low Priority': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top vendors
    st.subheader("🏆 Top Performing Vendors")
    
    top_vendors = summary.get('top_vendors', [])
    if top_vendors:
        top_vendor_data = []
        for vendor_name, vendor_data in top_vendors:
            top_vendor_data.append({
                'Vendor': vendor_name.replace('@', ''),
                'Lending Score': vendor_data['lending_score'],
                'Risk Category': vendor_data['risk_category'],
                'Avg Views': vendor_data['engagement']['avg_views'],
                'Posts/Week': vendor_data['posting_frequency']
            })
        
        top_df = pd.DataFrame(top_vendor_data)
        st.dataframe(top_df)
    
    # Detailed vendor analysis
    st.subheader("🔍 Detailed Vendor Analysis")
    
    detailed_analyses = vendor_results.get('detailed_analyses', {})
    valid_analyses = {k: v for k, v in detailed_analyses.items() if 'error' not in v}
    
    if valid_analyses:
        selected_vendor = st.selectbox(
            "Select vendor for detailed view:",
            list(valid_analyses.keys())
        )
        
        if selected_vendor:
            vendor_data = valid_analyses[selected_vendor]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Lending Score", f"{vendor_data['lending_score']:.1f}")
                st.metric("Risk Category", vendor_data['risk_category'])
                st.metric("Total Posts", vendor_data['total_posts'])
            
            with col2:
                st.metric("Avg Views/Post", f"{vendor_data['engagement']['avg_views']:.0f}")
                st.metric("Posts/Week", f"{vendor_data['posting_frequency']:.1f}")
                st.metric("Avg Price (ETB)", f"{vendor_data['price_metrics']['avg_price']:.0f}")
            
            # Top performing post
            top_post = vendor_data.get('top_performing_post', {})
            if top_post.get('text'):
                st.subheader("🌟 Top Performing Post")
                st.info(f"**Views:** {top_post['views']} | **Text:** {top_post['text']}")


def show_ner_playground():
    """Interactive NER testing interface"""
    st.header("🎮 NER Playground")
    
    st.markdown("Test the NER model with your own text!")
    
    # Check for available models
    models_dir = "models/"
    available_models = []
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it's a model directory
                if any(f.endswith('.json') for f in os.listdir(item_path)):
                    available_models.append(item_path)
    
    if not available_models:
        st.warning("No trained models found. Please run the training pipeline first.")
        return
    
    # Model selection
    selected_model = st.selectbox("Select model:", available_models)
    
    # Text input
    sample_texts = [
        "ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።",
        "አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር",
        "መርካቶ ውስጥ ጫማ 300 ብር",
        "Baby bottle for sale 150 birr in Bole area"
    ]
    
    text_input = st.text_area(
        "Enter text to analyze:",
        value=sample_texts[0],
        height=100,
        help="Enter Amharic or English text describing products, prices, and locations"
    )
    
    # Quick examples
    st.markdown("**Quick Examples:**")
    cols = st.columns(len(sample_texts))
    for i, (col, sample) in enumerate(zip(cols, sample_texts)):
        with col:
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.text_input = sample
                st.rerun()
    
    if st.button("🔍 Analyze Text", type="primary"):
        if text_input.strip():
            try:
                # Load model and analyze
                explainer = NERModelExplainer(selected_model)
                
                with st.spinner("Analyzing text..."):
                    predictions = explainer.pipeline(text_input)
                
                if predictions:
                    st.subheader("🎯 Detected Entities")
                    
                    for pred in predictions:
                        entity_type = pred['entity_group']
                        entity_text = pred['word']
                        confidence = pred['score']
                        
                        # Color code by entity type
                        if entity_type == 'Product':
                            color = 'blue'
                        elif entity_type == 'PRICE':
                            color = 'green'
                        elif entity_type == 'LOC':
                            color = 'orange'
                        else:
                            color = 'gray'
                        
                        st.markdown(
                            f"<span style='background-color: {color}; color: white; padding: 2px 6px; "
                            f"border-radius: 3px; margin: 2px;'>"
                            f"{entity_text} ({entity_type}: {confidence:.2f})"
                            f"</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No entities detected in the text.")
                
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")


def show_interpretability():
    """Show model interpretability analysis"""
    st.header("🔍 Model Interpretability")
    
    st.markdown("Understand how the NER model makes decisions using SHAP and LIME explanations.")
    
    # Check for interpretability results
    interpretability_dir = "models/interpretability/"
    
    if not os.path.exists(interpretability_dir):
        st.warning("No interpretability analysis found. Please run the interpretability pipeline first.")
        return
    
    # Load interpretability report
    report_path = os.path.join(interpretability_dir, "interpretability_report.json")
    
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        st.subheader("📊 Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Texts Analyzed", report.get('total_texts_analyzed', 0))
        
        with col2:
            difficult_cases = report.get('difficult_cases_analysis', {})
            st.metric("Difficult Cases", difficult_cases.get('difficult_cases', 0))
        
        with col3:
            easy_cases = difficult_cases.get('easy_cases', 0)
            st.metric("Easy Cases", easy_cases)
        
        # Confidence analysis
        if 'difficult_cases_analysis' in report:
            st.subheader("🎯 Confidence Analysis")
            
            confidence_dist = difficult_cases.get('confidence_distribution', [])
            if confidence_dist:
                fig = px.histogram(
                    x=confidence_dist,
                    nbins=20,
                    title="Distribution of Prediction Confidence",
                    labels={'x': 'Confidence Score', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # LIME explanations
        if 'lime_explanations' in report:
            st.subheader("🔬 LIME Explanations")
            
            lime_explanations = report['lime_explanations']
            for i, explanation in enumerate(lime_explanations):
                with st.expander(f"Difficult Case {i+1} (Confidence: {explanation['confidence']:.2f})"):
                    st.text(explanation['text'])
                    
                    if 'features' in explanation:
                        st.markdown("**Important Features:**")
                        for feature, importance in explanation['features'][:5]:
                            color = 'green' if importance > 0 else 'red'
                            st.markdown(f"- {feature}: {importance:.3f} <span style='color: {color}'>{'↑' if importance > 0 else '↓'}</span>", unsafe_allow_html=True)
    
    # Display confidence visualization if available
    confidence_plot_path = os.path.join(interpretability_dir, "confidence_analysis.png")
    if os.path.exists(confidence_plot_path):
        st.subheader("📈 Confidence Visualization")
        st.image(confidence_plot_path, caption="Model Confidence Analysis")


if __name__ == "__main__":
    main()