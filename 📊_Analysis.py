# streamlit_eda_simple.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Simple Customer Tier Analysis")

# --- Load Dataset ---
@st.cache_data
def load_data():
    data = pd.read_csv("data/synthetic_insurance_data.csv")
    return data

data = load_data()

st.title("🏆 Customer Tier Analysis")
st.write("Simple EDA for Insurance Customer Tiers")


# --- Sample Data Preview ---
st.subheader("📋 Data Preview")
st.dataframe(data.head(10))

# --- Basic Info ---
st.subheader("📊 Basic Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(data))
with col2:
    st.metric("Number of Features", len(data.columns))
with col3:
    st.metric("Target Column", "Tier")


# --- Tier Distribution ---
st.subheader("📈 Customer Tier Distribution")
tier_counts = data['Tier'].value_counts().reset_index()
tier_counts.columns = ['Tier', 'Count']

col1, col2 = st.columns(2)
with col1:
    fig = px.pie(tier_counts, values='Count', names='Tier',
                 color_discrete_sequence=px.colors.sequential.Mint, 
                 hole=0.3,
                 title="Customer Tier Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(tier_counts, x='Tier', y='Count', 
                 color='Tier',
                 color_discrete_map={'Bronze': "#A0E7D3", 'Silver': "#CCEBE5", 'Gold': "#56E2C9"},
                 title="Customer Tier Counts")
    st.plotly_chart(fig, use_container_width=True)




# --- Age Distribution by Tier ---
st.subheader("👥 Age Distribution by Tier")
fig = px.box(data, x='Tier', y='Age', color='Tier',
             color_discrete_sequence=px.colors.sequential.Mint)
st.plotly_chart(fig)

# --- Premium Distribution by Tier ---
st.subheader("💰 Annual Premium by Tier")
fig = px.box(data, x='Tier', y='Annual_Premium', color='Tier',
             color_discrete_sequence=px.colors.sequential.Mint)
st.plotly_chart(fig)

# --- Premium vs Vintage Scatter ---
st.subheader("📈 Annual Premium vs Vintage")
if 'Annual_Premium' in data.columns and 'Vintage' in data.columns:
    fig = px.scatter(data, x='Vintage', y='Annual_Premium', color='Tier',
                    color_discrete_map={'Bronze': '#CD7F32', 'Silver': '#C0C0C0', 'Gold': '#FFD700'},
                    title="Annual Premium vs Customer Vintage",
                    opacity=0.7)
    st.plotly_chart(fig)


# --- Age Group Analysis ---
st.subheader("👥 Age Group Analysis")
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                          labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])

age_tier = data.groupby(['Age_Group', 'Tier']).size().reset_index(name='Count')

fig = px.bar(age_tier, x='Age_Group', y='Count', color='Tier',
             color_discrete_map={'Bronze': "#B862E7", 'Silver': "#DFD4E7", 'Gold': "#CDA4E9"},
             barmode='stack',
             title="Age Group Distribution across Tiers")
st.plotly_chart(fig)


# --- Vehicle Analysis ---
st.subheader("🚗 Vehicle Analysis")
if 'Vehicle_Age' in data.columns and 'Vehicle_Damage' in data.columns:
    col1, col2 = st.columns(2)
    
    # with col1:
vehicle_age_tier = data.groupby(['Vehicle_Age', 'Tier']).size().reset_index(name='Count')
fig = px.bar(vehicle_age_tier, x='Vehicle_Age', y='Count', color='Tier',
            color_discrete_map={'Bronze': "#E65C5C", 'Silver': "#E2D9D9", 'Gold': "#E68B8B"},
            title="Vehicle Age by Tier")
st.plotly_chart(fig)

    # with col2:
vehicle_damage_tier = data.groupby(['Vehicle_Damage', 'Tier']).size().reset_index(name='Count')
fig = px.bar(vehicle_damage_tier, x='Vehicle_Damage', y='Count', color='Tier',
            color_discrete_map={'Bronze': "#E65C5C", 'Silver': "#E2D9D9", 'Gold': "#E68B8B"},
            title="Vehicle Damage History by Tier")
st.plotly_chart(fig)


num_features =  ['Age', 'Region_Code', 'Driving_License', 'Previously_Insured', 'Annual_Premium', 'Vintage']
# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")
num_features_existing = [col for col in num_features if col in data.columns]
if num_features_existing:
    corr = data[num_features_existing].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='ice'))
    fig.update_layout(title="Numerical Feature Correlation Heatmap")
    st.plotly_chart(fig, key="corr_heatmap")


# --- Customer Notes Analysis ---
st.subheader("💬 Customer Notes Analysis")

# Word cloud would be better but requires additional packages
# Showing sample notes instead
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Sample Gold Tier Notes:**")
    gold_notes = data[data['Tier'] == 'Gold']['Customer_Note'].head(5).tolist()
    for i, note in enumerate(gold_notes, 1):
        st.write(f"{i}. {note}")


with col2:
    st.write("**Sample Silver Tier Notes:**")
    silver_notes = data[data['Tier'] == 'Silver']['Customer_Note'].head(5).tolist()
    for i, note in enumerate(silver_notes, 1):
        st.write(f"{i}. {note}")

with col3:
    st.write("**Sample Bronze Tier Notes:**")
    bronze_notes = data[data['Tier'] == 'Bronze']['Customer_Note'].head(5).tolist()
    for i, note in enumerate(bronze_notes, 1):
        st.write(f"{i}. {note}")


st.markdown("----")

# --- Summary Statistics ---
st.subheader("📊 Summary Statistics by Tier")

tier_stats = data.groupby('Tier').agg({
    'Age': ['mean', 'std'],
    'Annual_Premium': ['mean', 'std'],
    'Vintage': ['mean', 'std'],
    'value_score': ['mean', 'std']
}).round(2)

# Flatten column names
tier_stats.columns = ['_'.join(col).strip() for col in tier_stats.columns.values]
tier_stats = tier_stats.reset_index()

st.dataframe(tier_stats.style.background_gradient(cmap='Purples'))

# --- Download Processed Data ---
st.subheader("📥 Download Processed Data")
if st.button("Download Analyzed Data as CSV"):
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="analyzed_insurance_data.csv",
        mime="text/csv"
    )
