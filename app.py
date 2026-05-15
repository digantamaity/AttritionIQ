import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AttritionIQ", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS (GLASSMORPHISM, FONTS, HIDDEN DEFAULTS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background Gradient for a premium feel */
    .stApp {
        background: linear-gradient(135deg, #0A0F1C 0%, #111A2C 100%);
        color: #E2E8F0;
    }
    
    /* Hide Default Streamlit Elements */
    header {display: none !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.2s ease-in-out;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 196, 180, 0.3);
    }
    
    .glass-card-accent-1 { border-left: 4px solid #00C4B4; }
    .glass-card-accent-2 { border-left: 4px solid #FF4B4B; }
    .glass-card-accent-3 { border-left: 4px solid #FACA2B; }
    .glass-card-accent-4 { border-left: 4px solid #0068C9; }
    
    .metric-title {
        color: #94A3B8;
        font-size: 13px;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 32px;
        font-weight: 700;
    }
    
    .main-title {
        color: #00C4B4;
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 18px;
        font-weight: 400;
        margin-bottom: 32px;
    }
    
    /* Custom Footer */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(10, 15, 28, 0.85);
        backdrop-filter: blur(8px);
        color: #64748B;
        text-align: center;
        padding: 12px 0;
        font-size: 13px;
        z-index: 100;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Form styling */
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 16px;
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
    st.error("Dataset not found. Please run the training script.")
    st.stop()

# --- HEADER & NAVIGATION ---
col_logo, col_nav = st.columns([1, 4])

with col_logo:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=180)
    else:
        st.markdown("<h2 style='color:#00C4B4; margin-top:10px;'>AttritionIQ</h2>", unsafe_allow_html=True)

with col_nav:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Analytics", "ML Models", "Live Prediction", "Insights"],
        icons=["house", "bar-chart-line", "cpu", "graph-up-arrow", "lightbulb"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "rgba(255,255,255,0.03)", "border": "1px solid rgba(255,255,255,0.05)", "border-radius": "8px"},
            "icon": {"color": "#94A3B8", "font-size": "16px"}, 
            "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px", "--hover-color": "rgba(255,255,255,0.05)", "color": "#E2E8F0"},
            "nav-link-selected": {"background-color": "#00C4B4", "color": "#0A0F1C", "font-weight": "600"},
        }
    )

st.markdown("---")

# --- GLOBAL FILTERS (Hidden in sidebar to keep UI clean) ---
with st.sidebar:
    st.subheader("Global Filters")
    dept_filter = st.multiselect("Department", options=df['Department'].unique(), default=df['Department'].unique())
    gender_filter = st.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    jobrole_filter = st.multiselect("Job Role", options=df['JobRole'].unique(), default=df['JobRole'].unique())
    age_range = st.slider("Age Range", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(18, 60))

filtered_df = df[
    (df['Department'].isin(dept_filter)) &
    (df['Gender'].isin(gender_filter)) &
    (df['JobRole'].isin(jobrole_filter)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1])
]

# --- PAGES ---

if selected == "Home":
    st.markdown('<div class="main-title">AttritionIQ Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Proactively identify flight risks and understand workforce dynamics using advanced Machine Learning.</div>', unsafe_allow_html=True)
    
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    total_emp = len(filtered_df)
    attrition_count = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
    attrition_rate = (attrition_count / total_emp) * 100 if total_emp > 0 else 0
    avg_age = filtered_df['Age'].mean() if total_emp > 0 else 0
    avg_salary = filtered_df['MonthlyIncome'].mean() if total_emp > 0 else 0
    
    with col1:
        st.markdown(f'''
        <div class="glass-card glass-card-accent-1">
            <div class="metric-title">Total Employees</div>
            <div class="metric-value">{total_emp}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="glass-card glass-card-accent-2">
            <div class="metric-title">Attrition Rate</div>
            <div class="metric-value">{attrition_rate:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="glass-card glass-card-accent-3">
            <div class="metric-title">Average Age</div>
            <div class="metric-value">{avg_age:.1f} yrs</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="glass-card glass-card-accent-4">
            <div class="metric-title">Avg Monthly Income</div>
            <div class="metric-value">${avg_salary:,.0f}</div>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown("### Dataset Overview")
    st.dataframe(filtered_df.head(10), use_container_width=True)

elif selected == "Analytics":
    st.markdown('<div class="main-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive exploration of workforce demographics and attrition drivers.</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Demographics", "Income & Satisfaction"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attrition by Department")
            dept_att = filtered_df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
            st.bar_chart(dept_att, use_container_width=True)
            
        with col2:
            st.subheader("Age Distribution")
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

    st.markdown("---")
    st.markdown("### Advanced Static Visualizations")
    with st.expander("View High-Resolution Auto-Generated Charts"):
        c1, c2 = st.columns(2)
        if os.path.exists('charts/correlation_heatmap.png'):
            c1.image('charts/correlation_heatmap.png', caption='Correlation Heatmap')
        if os.path.exists('charts/monthly_income_attrition.png'):
            c2.image('charts/monthly_income_attrition.png', caption='Monthly Income vs Attrition')

elif selected == "ML Models":
    st.markdown('<div class="main-title">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review predictive model performance and algorithm selection.</div>', unsafe_allow_html=True)
    
    if model_data:
        st.success(f"Best Model Loaded: **{model_data['model_name']}**")
        st.markdown("### Best Model Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = model_data['metrics']
        
        with col1: st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        with col2: st.metric("Precision", f"{metrics['Precision']:.2%}")
        with col3: st.metric("Recall", f"{metrics['Recall']:.2%}")
        with col4: st.metric("F1-Score", f"{metrics['F1-score']:.2%}")
        with col5: st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")
        
        st.markdown("---")
        st.markdown("### Model Comparison Chart")
        if os.path.exists('charts/model_comparison.png'):
            st.image('charts/model_comparison.png', use_column_width=True)
    else:
        st.error("Model data not found. Please run the training script.")

elif selected == "Live Prediction":
    st.markdown('<div class="main-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Input employee details to predict their likelihood of leaving the company.</div>', unsafe_allow_html=True)
    
    if not model_data:
        st.warning("Please train the model first by running the training script.")
        st.stop()
        
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.form("prediction_form"):
            st.markdown("### Employee Profile Form")
            
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
                
            submit = st.form_submit_button("Predict Attrition Risk", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
    if submit:
        input_dict = {}
        for col in df.columns:
            if col != 'Attrition':
                if df[col].dtype == 'object':
                    input_dict[col] = df[col].mode()[0]
                else:
                    input_dict[col] = df[col].median()
                    
        input_dict['Age'] = age
        input_dict['MonthlyIncome'] = monthly_income
        input_dict['OverTime'] = overtime
        input_dict['JobSatisfaction'] = job_satisfaction
        input_dict['YearsAtCompany'] = years_at_company
        input_dict['Department'] = department
        input_dict['JobRole'] = job_role
        input_dict['MaritalStatus'] = marital_status
        
        input_df = pd.DataFrame([input_dict])
        X_processed = model_data['preprocessor'].transform(input_df)
        prob = model_data['model'].predict_proba(X_processed)[0][1]
        
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prob > 0.5:
                st.error("### High Flight Risk Detected")
                st.markdown("This employee demonstrates patterns consistent with historical attrition profiles.")
            else:
                st.success("### Stable Retention Profile")
                st.markdown("This employee demonstrates patterns consistent with historical retention profiles.")
                
        with res_col2:
            st.metric("Risk Score (Probability)", f"{prob * 100:.1f}%")
            st.progress(float(prob))
            
        st.info("Note: The model uses default typical values for features not explicitly requested in the input form.")

elif selected == "Insights":
    st.markdown('<div class="main-title">Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Data-driven recommendations to reduce employee attrition and improve retention.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
    
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
    
    </div>
    """, unsafe_allow_html=True)
    
# --- FOOTER ---
st.markdown("""
<div class="custom-footer">
    Powered by AttritionIQ Machine Learning Engine | Secure HR Analytics Dashboard
</div>
""", unsafe_allow_html=True)
