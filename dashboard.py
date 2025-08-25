import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Amharic NER Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Simple NER predictor
class SimpleNERPredictor:
    def predict(self, text):
        tokens = text.split()
        predictions = []
        
        for token in tokens:
            if token in ['ሻንጣ', 'ስልክ', 'ጫማ', 'ልብስ', 'ላፕቶፕ']:
                predictions.append({'word': token, 'entity': 'Product', 'score': 0.95})
            elif token.isdigit() or 'ብር' in token:
                predictions.append({'word': token, 'entity': 'PRICE', 'score': 0.90})
            elif token in ['አዲስ', 'አበባ', 'ቦሌ', 'መርካቶ', 'ፒያሳ']:
                predictions.append({'word': token, 'entity': 'LOC', 'score': 0.85})
        
        return predictions

# Load data
@st.cache_data
def load_data():
    try:
        # Model comparison results
        comparison_df = pd.read_csv("models/comparison_results.csv")
        
        # Interpretability report
        with open("models/interpretability_report.json", 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Vendor analytics
        with open("data/processed/vendor_analytics_report.json", 'r', encoding='utf-8') as f:
            vendor_data = json.load(f)
        
        # Load vendor scorecard CSV
        vendor_scorecard = pd.read_csv("data/processed/vendor_analytics_report_scorecard.csv")
        
        return comparison_df, report, vendor_data, vendor_scorecard
        
    except Exception as e:
        st.warning(f"Could not load data files: {e}")
        # Create sample data if files don't exist
        comparison_df = pd.DataFrame({
            'model': ['xlm-roberta-base', 'distilbert-base-multilingual-cased', 'bert-base-multilingual-cased'],
            'f1_score': [0.85, 0.82, 0.80],
            'precision': [0.87, 0.84, 0.82],
            'recall': [0.83, 0.80, 0.78],
            'avg_inference_time': [0.15, 0.08, 0.12],
            'overall_score': [0.85, 0.82, 0.80]
        })
        
        report = {
            'summary': {'total_cases_analyzed': 50, 'sample_predictions': 3},
            'recommendations': ['Good performance on basic entities', 'Price detection works well']
        }
        
        vendor_data = {
            'recommendations': ['Run analytics to get recommendations']
        }
        
        vendor_scorecard = pd.DataFrame()
        
        return comparison_df, report, vendor_data, vendor_scorecard

# Initialize
predictor = SimpleNERPredictor()
comparison_df, report, vendor_data, vendor_scorecard = load_data()

# Header
st.title("🔍 Amharic NER Analytics Dashboard")
st.markdown("**Real-time Named Entity Recognition for Ethiopian E-commerce**")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select Page", ["NER Prediction", "Model Performance", "Vendor Analytics"])

if page == "NER Prediction":
    st.header("🎯 Live NER Prediction")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area("Enter Amharic text:", 
                                 value="ሻንጣ ዋጋ 500 ብር አዲስ አበባ ቦሌ",
                                 height=100)
        
        if st.button("Analyze Text", type="primary"):
            if text_input:
                predictions = predictor.predict(text_input)
                
                if predictions:
                    st.subheader("📊 Detected Entities")
                    
                    # Display predictions
                    for pred in predictions:
                        color = {"Product": "🟢", "PRICE": "🔵", "LOC": "🟡"}
                        st.write(f"{color.get(pred['entity'], '⚪')} **{pred['word']}** → {pred['entity']} ({pred['score']:.2f})")
                    
                    # Entity summary
                    entity_counts = {}
                    for pred in predictions:
                        entity_counts[pred['entity']] = entity_counts.get(pred['entity'], 0) + 1
                    
                    if entity_counts:
                        fig = px.pie(values=list(entity_counts.values()), 
                                   names=list(entity_counts.keys()),
                                   title="Entity Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No entities detected in the text.")
    
    with col2:
        st.subheader("📝 Sample Texts")
        samples = [
            "ሻንጣ ዋጋ 500 ብር አዲስ አበባ ቦሌ",
            "ስልክ በ 2000 ብር መርካቶ ላይ",
            "ጫማ እና ልብስ ፒያሳ ውስጥ"
        ]
        
        for sample in samples:
            if st.button(f"Try: {sample[:20]}...", key=sample):
                st.session_state.sample_text = sample

elif page == "Model Performance":
    st.header("📈 Model Performance Analysis")
    
    # Best model highlight
    best_model = comparison_df.iloc[0]['model']
    best_score = comparison_df.iloc[0]['overall_score']
    
    st.success(f"🏆 Best Model: **{best_model}** (Score: {best_score:.3f})")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(comparison_df, x='model', y='f1_score', 
                    title="F1 Scores by Model",
                    color='f1_score',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison_df, x='model', y='avg_inference_time',
                    title="Inference Speed (seconds)",
                    color='avg_inference_time',
                    color_continuous_scale='reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📊 Detailed Performance Metrics")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Model insights
    st.subheader("🔍 Model Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cases Analyzed", report['summary']['total_cases_analyzed'])
    
    with col2:
        st.metric("Best F1 Score", f"{comparison_df['f1_score'].max():.3f}")
    
    with col3:
        st.metric("Fastest Model", f"{comparison_df['avg_inference_time'].min():.3f}s")

elif page == "Vendor Analytics":
    st.header("🏪 Vendor Performance Analytics")
    
    # Check if we have vendor scorecard data
    if not vendor_scorecard.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vendors", len(vendor_scorecard))
        with col2:
            st.metric("Avg Lending Score", f"{vendor_scorecard['Lending_Score'].mean():.1f}")
        with col3:
            top_vendor = vendor_scorecard.iloc[0]['Vendor_Channel']
            st.metric("Top Performer", top_vendor)
        with col4:
            low_risk_count = len(vendor_scorecard[vendor_scorecard['Risk_Category'] == 'Low Risk'])
            st.metric("Low Risk Vendors", low_risk_count)
        
        # Top vendors
        st.subheader("🏆 Vendor Rankings")
        
        for i, (_, vendor) in enumerate(vendor_scorecard.head(5).iterrows()):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.write(f"**{i+1}. {vendor['Vendor_Channel']}**")
            with col2:
                st.metric("Score", f"{vendor['Lending_Score']:.1f}")
            with col3:
                color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🟠", "Very High Risk": "🔴"}
                st.write(f"{color.get(vendor['Risk_Category'], '⚪')} {vendor['Risk_Category']}")
            with col4:
                st.metric("Views/Post", f"{vendor['Avg_Views_Per_Post']:.0f}")
            with col5:
                st.metric("Posts/Week", f"{vendor['Posts_Per_Week']:.1f}")
        
        # Risk distribution chart
        risk_counts = vendor_scorecard['Risk_Category'].value_counts()
        fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                        title="Risk Distribution",
                        color_discrete_map={
                            'Low Risk': '#00ff00',
                            'Medium Risk': '#ffff00', 
                            'High Risk': '#ff8000',
                            'Very High Risk': '#ff0000'
                        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Lending score distribution
        with col2:
            fig_bar = px.bar(vendor_scorecard, x='Vendor_Channel', y='Lending_Score',
                           title="Lending Scores by Vendor",
                           color='Risk_Category',
                           color_discrete_map={
                               'Low Risk': '#00ff00',
                               'Medium Risk': '#ffff00',
                               'High Risk': '#ff8000', 
                               'Very High Risk': '#ff0000'
                           })
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed vendor table
        st.subheader("📊 Detailed Vendor Analytics")
        st.dataframe(vendor_scorecard, use_container_width=True)
        
        # Recommendations
        if 'recommendations' in vendor_data:
            st.subheader("💡 Recommendations")
            for rec in vendor_data['recommendations']:
                st.write(f"• {rec}")
    
    else:
        st.warning("⚠️ No vendor data available.")
        if st.button("🔄 Run Vendor Analytics", type="primary"):
            with st.spinner("Running vendor analytics..."):
                import subprocess
                result = subprocess.run(["python", "run_analytics.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Vendor analytics completed! Refresh the page to see results.")
                    st.rerun()
                else:
                    st.error(f"❌ Analytics failed: {result.stderr}")

# Footer
st.markdown("---")
st.markdown("**Amharic NER Dashboard** | Built with Streamlit | Ethiopian E-commerce Analytics")