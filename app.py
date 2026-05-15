import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="AttritionIQ Dashboard", page_icon="👥", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00C4B4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .metric-title {
        color: #A0AEC0;
        font-size: 14px;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 700;
    }
    .hero-title {
        color: #00C4B4;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        color: #E2E8F0;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stProgress .st-bo {
        background-color: #00C4B4;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA AND MODEL ---
DATA_FILE = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
MODEL_FILE = 'models/model.pkl'

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Clean basic useless columns for display
        cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    return None

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None

df = load_data()
model_data = load_model()

if df is None:
    st.error("⚠️ Dataset not found. Please run `python train_model.py` first to download data and train the model.")
    st.stop()

if model_data is None:
    st.warning("⚠️ Model not found. Running predictions will not work until you run `python train_model.py`.")

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.title("Navigation & Filters")

# Sidebar Navigation
page = st.sidebar.radio("Go to:", 
    ["🏠 Home / Hero", 
     "📊 Analytics Dashboard", 
     "🤖 Machine Learning", 
     "🔮 Live Prediction",
     "💡 Business Insights"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters (Analytics)")
dept_filter = st.sidebar.multiselect("Department", options=df['Department'].unique(), default=df['Department'].unique())
gender_filter = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
jobrole_filter = st.sidebar.multiselect("Job Role", options=df['JobRole'].unique(), default=df['JobRole'].unique())
age_range = st.sidebar.slider("Age Range", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(18, 60))

# Apply filters
filtered_df = df[
    (df['Department'].isin(dept_filter)) &
    (df['Gender'].isin(gender_filter)) &
    (df['JobRole'].isin(jobrole_filter)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1])
]

# --- PAGES ---

if page == "🏠 Home / Hero":
    st.markdown('<div class="hero-title">AttritionIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Proactively identify flight risks and understand workforce dynamics using Machine Learning.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to **AttritionIQ**. This end-to-end platform allows HR and Management to:
    - **Explore** past attrition trends.
    - **Predict** the likelihood of a specific employee leaving.
    - **Understand** key drivers of attrition.
    """)
    
    st.markdown("### 🏆 Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    
    total_emp = len(filtered_df)
    attrition_count = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
    attrition_rate = (attrition_count / total_emp) * 100 if total_emp > 0 else 0
    avg_age = filtered_df['Age'].mean() if total_emp > 0 else 0
    avg_salary = filtered_df['MonthlyIncome'].mean() if total_emp > 0 else 0
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Total Employees</div>
            <div class="metric-value">{total_emp}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-card" style="border-left-color: #FF4B4B;">
            <div class="metric-title">Attrition Rate</div>
            <div class="metric-value">{attrition_rate:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="metric-card" style="border-left-color: #FACA2B;">
            <div class="metric-title">Average Age</div>
            <div class="metric-value">{avg_age:.1f} yrs</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="metric-card" style="border-left-color: #0068C9;">
            <div class="metric-title">Avg Monthly Income</div>
            <div class="metric-value">${avg_salary:,.0f}</div>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown("### 🗃️ Dataset Overview")
    st.dataframe(filtered_df.head(10), use_container_width=True)

elif page == "📊 Analytics Dashboard":
    st.title("📊 AttritionIQ Analytics")
    st.write("Interactive exploration of workforce demographics and attrition drivers.")
    
    tab1, tab2 = st.tabs(["Demographics", "Income & Satisfaction"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attrition by Department")
            dept_att = filtered_df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
            st.bar_chart(dept_att, use_container_width=True)
            
        with col2:
            st.subheader("Age Distribution")
            # Interactive histogram via Streamlit native bar chart
            age_dist = filtered_df['Age'].value_counts().sort_index()
            st.bar_chart(age_dist, use_container_width=True)
            
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Average Monthly Income by Job Role")
            income_role = filtered_df.groupby('JobRole')['MonthlyIncome'].mean().sort_values(ascending=False)
            st.bar_chart(income_role, use_container_width=True)
        with col2:
            st.subheader("Attrition by OverTime")
            ot_att = filtered_df.groupby(['OverTime', 'Attrition']).size().unstack(fill_value=0)
            st.bar_chart(ot_att, use_container_width=True)

    st.markdown("### 📷 Advanced Static Visualizations")
    with st.expander("View High-Resolution Auto-Generated Charts"):
        c1, c2 = st.columns(2)
        if os.path.exists('charts/correlation_heatmap.png'):
            c1.image('charts/correlation_heatmap.png', caption='Correlation Heatmap')
        if os.path.exists('charts/monthly_income_attrition.png'):
            c2.image('charts/monthly_income_attrition.png', caption='Monthly Income vs Attrition')

elif page == "🤖 Machine Learning":
    st.title("🤖 Machine Learning Model Evaluation")
    st.write("Review how our predictive models performed and which model was selected for live prediction.")
    
    if model_data:
        st.success(f"✅ Best Model Loaded: **{model_data['model_name']}**")
        st.markdown("### 📈 Best Model Performance Metrics")
        m_cols = st.columns(5)
        m_cols[0].metric("Accuracy", f"{model_data['metrics']['Accuracy']:.2%}")
        m_cols[1].metric("Precision", f"{model_data['metrics']['Precision']:.2%}")
        m_cols[2].metric("Recall", f"{model_data['metrics']['Recall']:.2%}")
        m_cols[3].metric("F1-Score", f"{model_data['metrics']['F1-score']:.2%}")
        m_cols[4].metric("ROC-AUC", f"{model_data['metrics']['ROC-AUC']:.2%}")
        
        st.markdown("### 🏆 Model Comparison Chart")
        if os.path.exists('charts/model_comparison.png'):
            st.image('charts/model_comparison.png', use_column_width=True)
        else:
            st.info("Model comparison chart not found. Run training script.")
    else:
        st.error("Model data not found. Please run the training script.")

elif page == "🔮 Live Prediction":
    st.title("🔮 AttritionIQ Live Prediction")
    st.write("Input employee details to predict their likelihood of leaving the company.")
    
    if not model_data:
        st.warning("⚠️ Please train the model first by running `python train_model.py`.")
        st.stop()
        
    with st.form("prediction_form"):
        st.subheader("Employee Details")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
        with col2:
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
            department = st.selectbox("Department", df['Department'].unique())
        with col3:
            job_role = st.selectbox("Job Role", df['JobRole'].unique())
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        with col4:
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            marital_status = st.selectbox("Marital Status", df['MaritalStatus'].unique())
            
        submit = st.form_submit_button("Predict Attrition Risk 🚀")
        
    if submit:
        # Create a dataframe with default mode/median values for ALL features
        input_dict = {}
        for col in df.columns:
            if col != 'Attrition':
                if df[col].dtype == 'object':
                    input_dict[col] = df[col].mode()[0]
                else:
                    input_dict[col] = df[col].median()
                    
        # Update with user inputs
        input_dict['Age'] = age
        input_dict['MonthlyIncome'] = monthly_income
        input_dict['OverTime'] = overtime
        input_dict['JobSatisfaction'] = job_satisfaction
        input_dict['YearsAtCompany'] = years_at_company
        input_dict['Department'] = department
        input_dict['JobRole'] = job_role
        input_dict['MaritalStatus'] = marital_status
        
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess
        X_processed = model_data['preprocessor'].transform(input_df)
        
        # Predict
        prob = model_data['model'].predict_proba(X_processed)[0][1] # Probability of 'Yes'
        prediction = model_data['model'].predict(X_processed)[0]
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prob > 0.5:
                st.error("### 🚨 Employee Likely To Leave")
            else:
                st.success("### ✅ Employee Likely To Stay")
                
        with res_col2:
            st.metric("Confidence Percentage (Risk Score)", f"{prob * 100:.1f}%")
            st.progress(float(prob))
            
        st.info("ℹ️ **Note:** The model uses default typical values for features not explicitly requested in the input form.")

elif page == "💡 Business Insights":
    st.title("💡 Actionable Business Insights")
    st.write("Data-driven recommendations to reduce employee attrition and improve retention.")
    
    st.markdown("""
    Based on our comprehensive exploratory data analysis (EDA) and machine learning feature importance, here are the key insights:
    
    ### 1. Overtime is a Major Flight Risk
    - **Insight**: Employees who work overtime have a significantly higher attrition rate.
    - **Recommendation**: Review workload distribution. Implement mandatory rest periods or offer extra compensation/time-off in lieu for employees consistently working overtime.
    
    ### 2. Income Disparity and Turnover
    - **Insight**: Employees with lower monthly income (particularly below $3,500) show much higher tendencies to leave.
    - **Recommendation**: Conduct a market salary review for entry-level and junior roles. Consider restructuring compensation packages or introducing performance-based bonuses.
    
    ### 3. Departmental Stress
    - **Insight**: The Sales department frequently exhibits the highest attrition rates compared to R&D and HR.
    - **Recommendation**: Investigate the high-pressure environment in Sales. Introduce better support systems, achievable quotas, and mental health resources.
    
    ### 4. Career Stagnation
    - **Insight**: Employees who haven't had a promotion in the last 3-4 years are highly susceptible to leaving.
    - **Recommendation**: Develop clear career progression pathways. Ensure regular performance reviews and upskilling opportunities are available.
    
    ### 5. Age Factor
    - **Insight**: Younger employees (Age 18-25) have the highest turnover rates.
    - **Recommendation**: Enhance the onboarding process. Offer mentorship programs to better integrate young talent into the company culture.
    """)
    st.success("By addressing these key areas proactively, management can significantly lower the attrition rate and retain valuable talent.")
