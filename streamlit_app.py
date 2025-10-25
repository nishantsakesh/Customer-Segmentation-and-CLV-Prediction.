import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import datetime as dt
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(layout="wide", page_title="Customer Segmentation & CLV Dashboard")

# These functions are needed to process the uploaded data in real-time.

def preprocess_data(df_raw):
    """Cleans the raw transactional data."""
    df = df_raw.copy()
    
    # Handle missing CustomerIDs
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    
    # Handle cancelled orders and negative quantities
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Convert InvoiceDate
    try:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    except Exception as e:
        st.error(f"Error parsing InvoiceDate: {e}. Please ensure format is correct.")
        return None
        
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def create_rfm_features(df_processed, snapshot_date_str):
    """Aggregates data to customer-level RFM and other features."""
    
    snapshot_date = pd.to_datetime(snapshot_date_str)
    
    customer_features = df_processed.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum'),
        FirstPurchaseDate=('InvoiceDate', 'min'),
        NumUniqueProducts=('StockCode', 'nunique')
    ).reset_index()

    # Create Additional Features
    customer_features['AOV'] = customer_features['Monetary'] / customer_features['Frequency']
    customer_features['Tenure'] = (snapshot_date - customer_features['FirstPurchaseDate']).dt.days
    
    # Handle potential division by zero if tenure is 0
    customer_features['PurchaseFrequencyRate'] = customer_features['Frequency'] / (customer_features['Tenure'] + 1)
    
    # Ensure no negative recency
    customer_features['Recency'] = customer_features['Recency'].apply(lambda x: max(x, 0))

    return customer_features

def create_rfm_segments(rfm_df):
    """Creates RFM scores and segments."""
    df = rfm_df.copy()
    
    if df.empty or len(df) < 5:
        st.warning("Not enough data to create 5 quantile-based segments. Returning unsegmented data.")
        df['R_Score'] = 1
        df['F_Score'] = 1
        df['M_Score'] = 1
        df['RFM_Score'] = '111'
        df['Segment'] = 'Needs Attention'
        return df

    # Create scores
    df['R_Score'] = pd.qcut(df['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    df['M_Score'] = pd.qcut(df['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    
    df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
    
    # Define segment map
    segment_map = {
        r'[4-5][4-5][4-5]': 'Champions',
        r'[3-5][3-5][1-3]': 'Loyal Customers',
        r'[4-5][1-3][4-5]': 'Potential Loyalists',
        r'[4-5][1-3][1-3]': 'New Customers',
        r'[3-5][1-3][1-3]': 'Promising',
        r'[3-4][4-5][1-5]': 'Loyal Customers (Needs Attention)',
        r'[2-3][2-3][2-3]': 'Needs Attention',
        r'[2-3][1-5][4-5]': 'At Risk (High Spend, Slipping)',
        r'[1-2][3-5][3-5]': 'At Risk (High Value, Not Recent)',
        r'[1-2][1-2][3-5]': 'Hibernating (High Spend, Lost)',
        r'[1-2][1-5][1-2]': 'Lost'
    }
    
    df['Segment'] = df['RFM_Score'].replace(segment_map, regex=True)
    df['Segment'] = df['Segment'].apply(lambda x: 'Lost' if str(x).isdigit() else x)

    return df

# --- Model Loading ---
@st.cache_resource
def load_model_and_scaler():
    """Loads the pre-trained XGBoost model and scaler."""
    model_path = 'clv_model.pkl'
    scaler_path = 'scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"Error: Model or scaler file not found. \n"
                 f"Please run the `customer_segmentation_analysis.ipynb` notebook first to generate: \n"
                 f"1. `{model_path}`\n"
                 f"2. `{scaler_path}`")
        return None, None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

model, scaler = load_model_and_scaler()
FEATURES_FOR_CLV = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Tenure', 'NumUniqueProducts', 'PurchaseFrequencyRate']

# --- Main App UI ---
st.title("🛍️ Customer Segmentation & CLV Prediction Dashboard")
st.markdown("Upload your transactional data (CSV or Excel) to segment customers and predict their 3-month lifetime value.")

# --- Sidebar ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your transaction data", type=["csv", "xlsx"])

# Placeholder for processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

if uploaded_file is not None and model is not None:
    st.sidebar.info(f"File '{uploaded_file.name}' uploaded successfully.")
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
        
        with st.spinner("Processing data... This may take a moment."):
            # 1. Preprocess
            df_processed = preprocess_data(df_raw)
            
            if df_processed is not None and not df_processed.empty:
                # Set snapshot date
                snapshot_date = df_processed['InvoiceDate'].max() + dt.timedelta(days=1)
                st.sidebar.success(f"Snapshot date set to: {snapshot_date.strftime('%Y-%m-%d')}")

                # 2. Create RFM Features
                rfm_df = create_rfm_features(df_processed, snapshot_date.strftime('%Y-%m-%d'))

                # 3. Create RFM Segments
                rfm_df = create_rfm_segments(rfm_df)
                
                # 4. Predict CLV
                # Ensure all features are present
                for col in FEATURES_FOR_CLV:
                    if col not in rfm_df.columns:
                        st.error(f"Missing required feature: {col}")
                        st.stop()
                        
                features_to_scale = rfm_df[FEATURES_FOR_CLV]
                
                # Handle NaNs in features (e.g., AOV if Frequency is 0, though we filter)
                features_to_scale.fillna(0, inplace=True)
                
                scaled_features = scaler.transform(features_to_scale)
                
                # Predict log-transformed CLV
                log_clv = model.predict(scaled_features)
                
                # Inverse transform (expm1) and set negatives to 0
                rfm_df['Predicted_CLV_3M'] = np.expm1(log_clv)
                rfm_df['Predicted_CLV_3M'] = rfm_df['Predicted_CLV_3M'].apply(lambda x: max(0, x))
                
                st.session_state.processed_data = rfm_df.copy()
                st.sidebar.success("Data processed and CLV predicted!")
                
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.session_state.processed_data = None

elif model is None:
    st.info("Dashboard is waiting for model files to be loaded.")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")

# --- Main Dashboard Area ---
if st.session_state.processed_data is not None:
    rfm_data = st.session_state.processed_data
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧑‍💻 Customer Search", "🔮 Manual CLV Predictor", "💾 Full Data"])

    # --- Tab 1: Dashboard ---
    with tab1:
        st.header("Overall Customer Analysis")
        
        # KPIs
        total_customers = rfm_data['CustomerID'].nunique()
        total_revenue = rfm_data['Monetary'].sum()
        avg_predicted_clv = rfm_data['Predicted_CLV_3M'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Total Historical Revenue", f"£{total_revenue:,.2f}")
        col3.metric("Avg. Predicted 3M-CLV", f"£{avg_predicted_clv:,.2f}")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Segment Distribution
            st.subheader("Customer Segment Distribution")
            segment_counts = rfm_data['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            fig1 = px.bar(segment_counts, x='Segment', y='Count', color='Segment', 
                          title='Customer Segment Distribution')
            fig1.update_layout(xaxis={'categoryorder':'total descending'}, height=450)
            st.plotly_chart(fig1, use_container_width=True)
            
            # 2. Top 10 Customers by Predicted CLV
            st.subheader("Top 10 Customers by Predicted CLV")
            top_10_clv = rfm_data.nlargest(10, 'Predicted_CLV_3M')[['CustomerID', 'Segment', 'Monetary', 'Predicted_CLV_3M']]
            fig3 = px.bar(top_10_clv, x='CustomerID', y='Predicted_CLV_3M', color='Segment',
                          hover_data=['Monetary'], title="Top 10 Customers by Predicted 3-Month CLV")
            fig3.update_layout(height=450)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            # 3. Recency vs. Frequency Scatter
            st.subheader("Recency vs. Frequency")
            fig2 = px.scatter(rfm_data.sample(min(1000, len(rfm_data))), 
                              x='Recency', y='Frequency', color='Segment', 
                              title='Recency vs. Frequency by Segment (Sampled)', 
                              hover_data=['CustomerID', 'Monetary', 'Predicted_CLV_3M'],
                              log_y=True, log_x=True,
                              color_discrete_sequence=px.colors.qualitative.Plotly)
            fig2.update_layout(height=450)
            st.plotly_chart(fig2, use_container_width=True)

            # 4. Segment Summary Radar Chart
            st.subheader("Segment Characteristics (Normalized)")
            segment_summary = rfm_data.groupby('Segment').agg(
                Recency=('Recency', 'mean'),
                Frequency=('Frequency', 'mean'),
                Monetary=('Monetary', 'mean'),
                Predicted_CLV_3M=('Predicted_CLV_3M', 'mean')
            ).reset_index()

            # Normalize for radar chart
            scaler_radar = StandardScaler() # Use a local scaler
            metrics_to_plot = ['Recency', 'Frequency', 'Monetary', 'Predicted_CLV_3M']
            summary_scaled = segment_summary.copy()
            summary_scaled[metrics_to_plot] = scaler_radar.fit_transform(summary_scaled[metrics_to_plot])
            summary_scaled['Recency'] = -summary_scaled['Recency'] # Invert recency
            
            fig4 = go.Figure()
            for i, row in summary_scaled.iterrows():
                fig4.add_trace(go.Scatterpolar(
                    r=row[metrics_to_plot].values.tolist() + [row[metrics_to_plot].values[0]],
                    theta=metrics_to_plot + [metrics_to_plot[0]],
                    fill='toself',
                    name=row['Segment']
                ))
            fig4.update_layout(
                title="Segment Profiles (Recency is inverted: higher is better)",
                height=450
            )
            st.plotly_chart(fig4, use_container_width=True)

    # --- Tab 2: Customer Search ---
    with tab2:
        st.header("Single Customer Lookup")
        customer_id_list = rfm_data['CustomerID'].unique().tolist()
        selected_customer = st.selectbox("Select a CustomerID", options=customer_id_list)
        
        if selected_customer:
            customer_data = rfm_data[rfm_data['CustomerID'] == selected_customer].iloc[0]
            st.subheader(f"Profile for Customer: {selected_customer}")
            
            st.markdown(f"### **Segment: {customer_data['Segment']}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted 3M-CLV", f"£{customer_data['Predicted_CLV_3M']:.2f}")
            col2.metric("Historical Total Spend", f"£{customer_data['Monetary']:.2f}")
            col3.metric("RFM Score", customer_data['RFM_Score'])
            
            st.divider()
            st.subheader("Detailed Metrics")
            st.dataframe(customer_data[FEATURES_FOR_CLV + ['Segment', 'Predicted_CLV_3M']].to_frame().T)

    # --- Tab 3: Manual CLV Predictor ---
    with tab3:
        st.header("Predict CLV for a New/Manual Customer Profile")
        st.markdown("Enter customer features to get an on-the-fly 3-month CLV prediction.")
        
        with st.form("clv_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                recency = st.number_input("Recency (days)", min_value=0, value=30)
                frequency = st.number_input("Frequency (total purchases)", min_value=1, value=5)
                monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, format="%.2f")
            with col2:
                aov = st.number_input("Average Order Value (AOV)", min_value=0.0, value=monetary/frequency, format="%.2f", disabled=True)
                tenure = st.number_input("Tenure (days since first purchase)", min_value=0, value=180)
            with col3:
                num_unique = st.number_input("Number of Unique Products", min_value=1, value=10)
                purchase_freq = st.number_input("Purchase Freq. Rate", min_value=0.0, value=frequency/(tenure+1), format="%.6f", disabled=True)
            
            submitted = st.form_submit_button("Predict CLV")
            
            if submitted:
                # Recalculate AOV and Freq Rate
                aov_calc = monetary / frequency if frequency > 0 else 0
                freq_rate_calc = frequency / (tenure + 1)
                
                # Create input DataFrame
                input_data = pd.DataFrame([[
                    recency, frequency, monetary, aov_calc, tenure, num_unique, freq_rate_calc
                ]], columns=FEATURES_FOR_CLV)
                
                # Scale features
                input_scaled = scaler.transform(input_data)
                
                # Predict
                log_clv_pred = model.predict(input_scaled)
                clv_pred = np.expm1(log_clv_pred)[0]
                clv_pred = max(0, clv_pred) # Ensure no negative
                
                st.success(f"**Predicted 3-Month CLV:**")
                st.subheader(f"£{clv_pred:,.2f}")

    # --- Tab 4: Full Data ---
    with tab4:
        st.header("Full Segmented Customer Data")
        st.dataframe(rfm_data)
        
        # Download button
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(rfm_data)
        
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="customer_segments_clv.csv",
            mime="text/csv",
        )
