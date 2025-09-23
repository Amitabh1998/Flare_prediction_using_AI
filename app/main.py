import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from app.model import train_all_models, predict_flare
from app.data import load_data

# Set page config
st.set_page_config(layout="wide", page_title="Sleep Flare Predictor", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
    body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; }
    .stButton>button { background-color: #3b82f6; color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; font-weight: 600; transition: background-color 0.3s ease; }
    .stButton>button:hover { background-color: #2563eb; }
    .stSlider label, .stRadio label { font-size: 1rem; font-weight: 600; color: #1f2937; margin-bottom: 0.5rem; }
    .card { background-color: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; }
    .prediction-box { background-color: #e8f4f8; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #b3d4fc; margin-top: 1rem; }
    .warning-box { background-color: #fef3c7; padding: 1rem; border-radius: 0.5rem; border: 1px solid #fcd34d; color: #78350f; margin-top: 1rem; }
    .stPlotlyChart { background-color: white; border-radius: 0.75rem; padding: 1rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
    h1 { color: #1f2937; font-size: 2.25rem; font-weight: 800; margin-bottom: 1rem; }
    h2 { color: #374151; font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; }
    .tooltip { position: relative; display: inline-block; cursor: pointer; }
    .tooltip .tooltiptext { visibility: hidden; width: 200px; background-color: #374151; color: white; text-align: center; border-radius: 0.5rem; padding: 0.75rem; position: absolute; z-index: 1; bottom: 125%; left: 50%; transform: translateX(-50%); opacity: 0; transition: opacity 0.3s; }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

# Load data
df = load_data()
features = ['pain', 'fatigue', 'sleep_hours', 'sleep_efficiency', 'pain_fatigue_interaction', 'mood_pain_interaction', 'pain_rolling_mean']

# Initialize model and training
model_path = 'models/flare_model_v3.pt'
model, model_type, accuracy, report, mean, std = train_all_models(df, features, model_path)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div class='card'>
            <h2 class='text-lg font-bold text-gray-800'>Sleep Flare Predictor</h2>
            <p class='text-gray-600'>A machine learning tool to predict flare-ups based on sleep and symptom data.</p>
            <p class='text-sm text-gray-500 mt-4'>Built with <i class='fas fa-heart text-red-500'></i> by xAI</p>
        </div>
        <div class='card'>
            <h3 class='text-md font-semibold text-gray-700'>Navigation</h3>
            <p class='text-gray-600'>Use the tabs to explore data, visualize trends, or predict flares.</p>
        </div>
    """, unsafe_allow_html=True)

# App title
st.markdown("""
    <div class='card'>
        <h1><i class='fas fa-moon mr-2'></i>Sleep Flare Predictor</h1>
        <p class='text-gray-600'>Explore sleep and symptom data to predict flare-ups with our intuitive, AI-powered tool.</p>
    </div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "📈 Visualizations", "🤖 Predict Flare"])

with tab1:
    st.markdown("<h2><i class='fas fa-table mr-2'></i>Data Exploration</h2>", unsafe_allow_html=True)
    st.markdown("<h3 class='text-lg font-semibold text-gray-700'>Dataset Overview</h3>", unsafe_allow_html=True)
    st.dataframe(df.style.set_table_styles([{'selector': 'th', 'props': [('background-color', '#1f2937'), ('color', 'white'), ('font-weight', '600')]}]), use_container_width=True)
    st.markdown("<h3 class='text-lg font-semibold text-gray-700'>Descriptive Statistics</h3>", unsafe_allow_html=True)
    st.dataframe(df[features + ['flare_next_day']].describe().style.format("{:.2f}"), use_container_width=True)
    st.markdown("<h3 class='text-lg font-semibold text-gray-700'>Flare Distribution</h3>", unsafe_allow_html=True)
    flare_counts = df['flare_next_day'].value_counts().rename({0: "No Flare", 1: "Flare"})
    fig = px.bar(x=flare_counts.index, y=flare_counts.values, color=flare_counts.index, color_discrete_map={0: '#3b82f6', 1: '#ef4444'}, labels={'x': 'Flare Status', 'y': 'Count'})
    fig.update_layout(showlegend=False, xaxis_title="Flare Status", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<h3 class='text-lg font-semibold text-gray-700'>Model Evaluation</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='p-4 bg-white rounded-lg shadow'>{report}</div>", unsafe_allow_html=True)
    if "failed to predict flares" in report:
        st.markdown('<div class="warning-box"><i class="fas fa-exclamation-triangle mr-2"></i>The model is predicting only "No Flare". Try retraining or adjusting inputs.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("<h2><i class='fas fa-chart-line mr-2'></i>Visualizations</h2>", unsafe_allow_html=True)
    st.markdown("<h3 class='text-lg font-semibold text-gray-700'>Correlation Heatmap</h3>", unsafe_allow_html=True)
    corr = df[features + ['flare_next_day']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    st.pyplot(fig, use_container_width=True)
    st.markdown("<h3 class='text-lg font-semibold text-gray-700'>Pain Trend by Subject</h3>", unsafe_allow_html=True)
    select_subject = st.selectbox("Select Subject", df['subject'].unique(), key="subject_select", help="Choose a subject to view their pain trend over time")
    subject_df = df[df['subject'] == select_subject]
    fig = px.line(subject_df, x='day', y='pain', title=f"Pain Trend for {select_subject}", markers=True, color_discrete_sequence=['#1f2937'])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<h3 class='text-lg font-semibold text-gray-700'>Average Metrics by Flare</h3>", unsafe_allow_html=True)
    avg_df = df.groupby('flare_next_day')[features].mean().reset_index()
    melt_df = pd.melt(avg_df, id_vars=['flare_next_day'], var_name='Metric', value_name='Value')
    fig = px.bar(melt_df, x='Metric', y='Value', color='flare_next_day', barmode='group', color_discrete_sequence=['#3b82f6', '#ef4444'])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2><i class='fas fa-robot mr-2'></i>Predict Flare-Up</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-gray-600'>Enter your sleep and symptom data to predict the likelihood of a flare-up tomorrow.</p>", unsafe_allow_html=True)
    with st.expander("Enter Your Data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 class='text-md font-semibold text-gray-700'>Symptom Metrics</h3>", unsafe_allow_html=True)
            pain = st.slider("Pain Level (0-10)", min_value=0, max_value=10, value=5, help="Self-reported pain level (0 = none, 10 = severe)", key="pain")
            st.markdown(f"<p class='text-sm text-gray-500'>Selected: {pain}</p>", unsafe_allow_html=True)
            fatigue = st.slider("Fatigue Level (0-10)", min_value=0, max_value=10, value=5, help="Self-reported fatigue level (0 = none, 10 = extreme)", key="fatigue")
            st.markdown(f"<p class='text-sm text-gray-500'>Selected: {fatigue}</p>", unsafe_allow_html=True)
            mood = st.slider("Mood Level (0-10)", min_value=0, max_value=10, value=5, help="Self-reported mood (0 = poor, 10 = excellent)", key="mood")
            st.markdown(f"<p class='text-sm text-gray-500'>Selected: {mood}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h3 class='text-md font-semibold text-gray-700'>Sleep Metrics</h3>", unsafe_allow_html=True)
            total_sleep = st.slider("Total Sleep (hours)", min_value=5.0, max_value=9.0, value=6.5, step=0.1, help="Total hours of sleep in a night", key="total_sleep", format="%.1f")
            st.markdown(f"<p class='text-sm text-gray-500'>Selected: {total_sleep:.1f} hours</p>", unsafe_allow_html=True)
            wake = st.slider("Wake Time (hours)", min_value=15.0, max_value=19.0, value=17.0, step=0.1, help="Hours spent awake during sleep period", key="wake", format="%.1f")
            st.markdown(f"<p class='text-sm text-gray-500'>Selected: {wake:.1f} hours</p>", unsafe_allow_html=True)
            sleep_hours = st.slider("Sleep Hours (normalized)", min_value=0.1, max_value=1.2, value=0.5, step=0.01, help="Normalized sleep duration metric", key="sleep_hours", format="%.2f")
            st.markdown(f"<p class='text-sm text-gray-500'>Selected: {sleep_hours:.2f}</p>", unsafe_allow_html=True)
            pain_rolling_mean = st.slider("Pain Rolling Mean (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, help="3-day average pain level", key="pain_rolling_mean", format="%.1f")
            st.markdown(f"<p class='text-sm text-gray-500'>Selected: {pain_rolling_mean:.1f}</p>", unsafe_allow_html=True)
        # Compute derived features
        sleep_efficiency = total_sleep / (total_sleep + wake) if (total_sleep + wake) > 0 else 0.0
        pain_fatigue_interaction = pain * fatigue
        mood_pain_interaction = mood * pain
        st.markdown(f"<p class='text-sm text-gray-500'>Computed Sleep Efficiency: {sleep_efficiency:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='text-sm text-gray-500'>Computed Pain-Fatigue Interaction: {pain_fatigue_interaction:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='text-sm text-gray-500'>Computed Mood-Pain Interaction: {mood_pain_interaction:.2f}</p>", unsafe_allow_html=True)
        if st.button("Predict Flare", use_container_width=True):
            with st.spinner("Predicting..."):
                input_data = np.array([[pain, fatigue, sleep_hours, sleep_efficiency, pain_fatigue_interaction, mood_pain_interaction, pain_rolling_mean]])
                prediction, prob_flare, output = predict_flare(model, input_data, model_type, mean, std)
                result_color = "#ef4444" if prediction == "Flare Likely" else "#10b981"
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h3 class="text-lg font-semibold" style="color: {result_color}">
                            <i class="fas {'fa-exclamation-circle' if prediction == 'Flare Likely' else 'fa-check-circle'} mr-2"></i>
                            Prediction: {prediction}
                        </h3>
                        <p class="text-gray-600">Probability of Flare: {prob_flare*100:.2f}%</p>
                        <p class="text-sm text-gray-500">Raw Output: {output if output is not None else 'N/A'}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if prediction == "No Flare Expected":
                    st.markdown('<div class="warning-box"><i class="fas fa-exclamation-triangle mr-2"></i>Model predicted "No Flare". If this persists for all inputs, the model may need retraining.</div>', unsafe_allow_html=True)