# streamlit_model_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Comparison Dashboard")

st.title("📊 Model Comparison Dashboard")
st.write("This dashboard visualizes the performance of all trained models.")

# --- Load Model Comparison CSV ---
@st.cache_data
def load_comparison():
    return pd.read_csv("data/model_evaluation_results.csv")

results_df = load_comparison()

st.subheader("Model Performance Table")
st.dataframe(results_df)

# --- Accuracy Bar Chart ---
st.subheader("Accuracy Comparison")
fig_acc = px.bar(results_df, x='model_name', y='accuracy', 
                 color='accuracy', color_continuous_scale='Mint',
                 text='accuracy', title="Model Accuracy Comparison")
fig_acc.update_traces(texttemplate='%{text:.2f}', textposition='outside')
st.plotly_chart(fig_acc, use_container_width=True)

# --- Precision, Recall, F1 Comparison ---
st.subheader("Precision, Recall, F1-Score Comparison")
metrics_df = results_df.melt(id_vars='model_name', value_vars=['precision','recall','f1_score'],
                             var_name='Metric', value_name='Score')

fig_metrics = px.bar(metrics_df, x='model_name', y='Score', color='Metric', barmode='group',
                     title="Precision, Recall, and F1-Score Comparison")
st.plotly_chart(fig_metrics, use_container_width=True)

# --- Best Model Info ---
best_model = results_df.loc[results_df['accuracy'].idxmax()]
st.success(f"🏆 Best Model: **{best_model['model_name']}** with Accuracy: **{best_model['accuracy']:.2f}**")
st.write("Metrics of Best Model:")
st.write(best_model)

# --- Optional: Select Metric to Visualize ---
st.subheader("Interactive Metric Comparison")
metric_option = st.selectbox("Select metric to visualize:", ['accuracy','precision','recall','f1_score'])
fig = px.bar(results_df, x='model_name', y=metric_option, 
             color=metric_option, color_continuous_scale='Pinkyl',
             text=metric_option, title=f"{metric_option} Comparison Across Models")
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)
