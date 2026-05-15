import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="AttritionIQ", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0A0F1C 0%, #111A2C 100%);
        color: #E2E8F0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    .block-container {padding-top: 2rem !important;}
    
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
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA AND DEFAULT MODEL ---
DATA_FILE = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
MODEL_FILE = 'models/model.pkl'

@st.cache_data
def load_default_data():
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

# --- AUTO-ML TRAINING FUNCTION ---
def train_dynamic_model(df, target_col):
    df_clean = df.dropna(subset=[target_col]).copy()
    y = df_clean[target_col]
    
    if y.nunique() > 10:
        raise ValueError(f"Target column '{target_col}' has {y.nunique()} unique values. Please select a categorical column (like 'Attrition' or 'left') to train a classifier.")
        
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    is_binary = len(le.classes_) == 2
    
    X = df_clean.drop(columns=[target_col])
    
    # Prevent OOM Crash: Filter out high-cardinality categorical columns (e.g., Emp ID, Names)
    cat_cols = []
    for col in X.select_dtypes(include=['object', 'category']).columns:
        if X[col].nunique() <= 50:
            cat_cols.append(col)
            
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Keep only selected columns to avoid massive OneHotEncoding
    X = X[num_cols + cat_cols]
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {'Accuracy': accuracy_score(y_test, y_pred)}
    
    if is_binary:
        metrics['Precision'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['F1-score'] = f1_score(y_test, y_pred, zero_division=0)
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['ROC-AUC'] = None
    else:
        metrics['Precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['F1-score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['ROC-AUC'] = None
        
    return {
        'pipeline': pipeline,
        'metrics': metrics,
        'target_encoder': le,
        'is_binary': is_binary,
        'feature_names': X.columns.tolist(),
        'target_classes': le.classes_.tolist()
    }

# --- SIDEBAR: DATA SOURCE & FILTERS ---
with st.sidebar:
    # Ensure data directory exists for persistence
    if not os.path.exists("data"):
        os.makedirs("data")

    # Read preference from URL with safety fallback
    try:
        source_param = st.query_params.get("source", "ibm")
        default_idx = 1 if source_param == "custom" else 0
    except Exception:
        default_idx = 0
        
    data_source = st.radio("Select Dataset", ["Sample IBM HR Dataset", "Upload Custom CSV"], index=default_idx)
    
    # Update URL params safely
    try:
        if data_source == "Upload Custom CSV":
            st.query_params["source"] = "custom"
        else:
            st.query_params["source"] = "ibm"
    except Exception:
        pass
    
    df = None
    CUSTOM_DATA_PATH = "data/custom_dataset.csv"
    
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df.to_csv(CUSTOM_DATA_PATH, index=False)
                
                if 'uploaded_filename' not in st.session_state or st.session_state['uploaded_filename'] != uploaded_file.name:
                    st.session_state['uploaded_filename'] = uploaded_file.name
                    st.session_state['custom_model_data'] = None
                    st.session_state['custom_target_col'] = None
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
                st.stop()
        elif os.path.exists(CUSTOM_DATA_PATH):
            try:
                df = pd.read_csv(CUSTOM_DATA_PATH)
                st.success("Loaded previously uploaded dataset!")
            except Exception:
                st.warning("Previous custom dataset found but could not be loaded. Please re-upload.")
                st.stop()
        else:
            st.info("Please upload a CSV file to view analytics.")
            st.stop()
    else:
        df = load_default_data()
        if df is None:
            st.error("Default dataset not found. Please run the training script.")
            st.stop()

    st.markdown("---")
    st.markdown("### Global Filters")
    
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() <= 50]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    filter_dict = {}
    
    if data_source == "Sample IBM HR Dataset":
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
    else:
        if cat_cols:
            selected_cat_filters = st.multiselect("Select Categorical Columns to Filter By", cat_cols, default=cat_cols[:min(2, len(cat_cols))])
            for col in selected_cat_filters:
                unique_vals = df[col].dropna().unique().tolist()
                filter_dict[col] = st.multiselect(f"Filter {col}", options=unique_vals, default=unique_vals)
                
        if num_cols:
            selected_num_filter = st.selectbox("Select Numerical Column to Filter By", ["None"] + num_cols)
            if selected_num_filter != "None":
                min_val = float(df[selected_num_filter].min())
                max_val = float(df[selected_num_filter].max())
                if min_val < max_val:
                    filter_dict[selected_num_filter] = st.slider(f"Range for {selected_num_filter}", min_val, max_val, (min_val, max_val))
                    
        filtered_df = df.copy()
        for col, filter_val in filter_dict.items():
            if col in cat_cols:
                filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
            elif col in num_cols:
                filtered_df = filtered_df[(filtered_df[col] >= filter_val[0]) & (filtered_df[col] <= filter_val[1])]

default_model_data = load_model()

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

# --- PAGES ---

if selected == "Home":
    st.markdown('<div class="main-title">AttritionIQ Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced Machine Learning and Data Analytics Engine.</div>', unsafe_allow_html=True)
    
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    if data_source == "Sample IBM HR Dataset":
        total_emp = len(filtered_df)
        attrition_count = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
        attrition_rate = (attrition_count / total_emp) * 100 if total_emp > 0 else 0
        avg_age = filtered_df['Age'].mean() if total_emp > 0 else 0
        avg_salary = filtered_df['MonthlyIncome'].mean() if total_emp > 0 else 0
        
        with col1: st.markdown(f'''<div class="glass-card glass-card-accent-1"><div class="metric-title">Total Employees</div><div class="metric-value">{total_emp}</div></div>''', unsafe_allow_html=True)
        with col2: st.markdown(f'''<div class="glass-card glass-card-accent-2"><div class="metric-title">Attrition Rate</div><div class="metric-value">{attrition_rate:.1f}%</div></div>''', unsafe_allow_html=True)
        with col3: st.markdown(f'''<div class="glass-card glass-card-accent-3"><div class="metric-title">Average Age</div><div class="metric-value">{avg_age:.1f} yrs</div></div>''', unsafe_allow_html=True)
        with col4: st.markdown(f'''<div class="glass-card glass-card-accent-4"><div class="metric-title">Avg Monthly Income</div><div class="metric-value">${avg_salary:,.0f}</div></div>''', unsafe_allow_html=True)
    else:
        total_rows = len(filtered_df)
        total_cols = len(filtered_df.columns)
        missing_vals = filtered_df.isnull().sum().sum()
        num_cat = len(cat_cols)
        
        with col1: st.markdown(f'''<div class="glass-card glass-card-accent-1"><div class="metric-title">Total Rows</div><div class="metric-value">{total_rows}</div></div>''', unsafe_allow_html=True)
        with col2: st.markdown(f'''<div class="glass-card glass-card-accent-2"><div class="metric-title">Total Columns</div><div class="metric-value">{total_cols}</div></div>''', unsafe_allow_html=True)
        with col3: st.markdown(f'''<div class="glass-card glass-card-accent-3"><div class="metric-title">Categorical Features</div><div class="metric-value">{num_cat}</div></div>''', unsafe_allow_html=True)
        with col4: st.markdown(f'''<div class="glass-card glass-card-accent-4"><div class="metric-title">Total Missing Values</div><div class="metric-value">{missing_vals}</div></div>''', unsafe_allow_html=True)
        
    st.markdown("### Dataset Overview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

elif selected == "Analytics":
    st.markdown('<div class="main-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive data exploration and dynamic visualizations.</div>', unsafe_allow_html=True)
    
    if data_source == "Sample IBM HR Dataset":
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
    else:
        st.markdown("### Dynamic Data Explorer")
        st.markdown("Select columns below to generate real-time charts based on your custom dataset.")
        
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox("Select X-Axis", df.columns)
        with c2:
            y_axis = st.selectbox("Select Y-Axis (Numerical preferred)", num_cols if num_cols else df.columns)
            
        st.markdown("#### Scatter Plot")
        if len(filtered_df) > 0:
            st.scatter_chart(filtered_df, x=x_axis, y=y_axis)
        else:
            st.info("No data available with current filters.")
            
        if cat_cols:
            st.markdown("---")
            st.markdown("#### Categorical Distribution")
            cat_select = st.selectbox("Select Categorical Column to Count", cat_cols)
            st.bar_chart(filtered_df[cat_select].value_counts())

elif selected == "ML Models":
    st.markdown('<div class="main-title">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review predictive model performance and algorithm selection.</div>', unsafe_allow_html=True)
    
    if data_source == "Sample IBM HR Dataset":
        if default_model_data:
            st.success(f"Best Model Loaded: **{default_model_data['model_name']}**")
            st.markdown("### Best Model Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            metrics = default_model_data['metrics']
            with col1: st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
            with col2: st.metric("Precision", f"{metrics['Precision']:.2%}")
            with col3: st.metric("Recall", f"{metrics['Recall']:.2%}")
            with col4: st.metric("F1-Score", f"{metrics['F1-score']:.2%}")
            with col5: st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")
            st.markdown("---")
            if os.path.exists('charts/model_comparison.png'):
                st.image('charts/model_comparison.png', use_column_width=True)
        else:
            st.error("Default model data not found.")
    else:
        st.markdown("### Auto-ML Training Engine")
        st.info("Train a custom Machine Learning model on your uploaded dataset instantly.")
        
        target_col = st.selectbox("Select Target Variable to Predict:", df.columns)
        
        if st.button("Train Custom AI Model", type="primary"):
            with st.spinner(f"Training Random Forest model to predict '{target_col}'..."):
                try:
                    custom_model_data = train_dynamic_model(df, target_col)
                    st.session_state['custom_model_data'] = custom_model_data
                    st.session_state['custom_target_col'] = target_col
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    
        if st.session_state.get('custom_model_data'):
            st.markdown("---")
            st.markdown("### Custom Model Performance Metrics")
            metrics = st.session_state['custom_model_data']['metrics']
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
            with c2: st.metric("Precision", f"{metrics['Precision']:.2%}")
            with c3: st.metric("Recall", f"{metrics['Recall']:.2%}")
            with c4: st.metric("F1-Score", f"{metrics['F1-score']:.2%}")

elif selected == "Live Prediction":
    st.markdown('<div class="main-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Input specific details to generate real-time AI predictions.</div>', unsafe_allow_html=True)
    
    if data_source == "Sample IBM HR Dataset":
        if not default_model_data:
            st.warning("Please train the IBM model first by running the training script.")
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
                    if pd.api.types.is_numeric_dtype(df[col]):
                        input_dict[col] = df[col].median()
                    else:
                        input_dict[col] = df[col].mode()[0]
                        
            input_dict['Age'] = age
            input_dict['MonthlyIncome'] = monthly_income
            input_dict['OverTime'] = overtime
            input_dict['JobSatisfaction'] = job_satisfaction
            input_dict['YearsAtCompany'] = years_at_company
            input_dict['Department'] = department
            input_dict['JobRole'] = job_role
            input_dict['MaritalStatus'] = marital_status
            
            input_df = pd.DataFrame([input_dict])
            X_processed = default_model_data['preprocessor'].transform(input_df)
            prob = default_model_data['model'].predict_proba(X_processed)[0][1]
            
            st.markdown("---")
            st.markdown("### Prediction Result")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if prob > 0.5:
                    st.error("### High Flight Risk Detected")
                else:
                    st.success("### Stable Retention Profile")
            with res_col2:
                st.metric("Risk Score (Probability)", f"{prob * 100:.1f}%")
                st.progress(float(prob))
    else:
        # Dynamic Prediction for Custom Dataset
        if not st.session_state.get('custom_model_data'):
            st.warning("⚠️ Please select a target variable and train a custom model in the **ML Models** tab first!")
            st.stop()
            
        custom_model = st.session_state['custom_model_data']
        target_col = st.session_state['custom_target_col']
        
        st.markdown(f"### Predict '{target_col}'")
        
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            with st.form("custom_predict_form"):
                st.markdown("### Custom Input Form")
                input_dict = {}
                cols = st.columns(3)
                
                # Dynamically generate inputs for all features
                for idx, col_name in enumerate(custom_model['feature_names']):
                    with cols[idx % 3]:
                        if pd.api.types.is_numeric_dtype(df[col_name]):
                            med_val = float(df[col_name].median())
                            input_dict[col_name] = st.number_input(col_name, value=med_val)
                        else:
                            unique_vals = df[col_name].dropna().unique().tolist()
                            input_dict[col_name] = st.selectbox(col_name, unique_vals)
                            
                submit = st.form_submit_button("Generate AI Prediction", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
            
        if submit:
            input_df = pd.DataFrame([input_dict])
            pipeline = custom_model['pipeline']
            pred_encoded = pipeline.predict(input_df)[0]
            pred_label = custom_model['target_encoder'].inverse_transform([pred_encoded])[0]
            
            st.markdown("---")
            st.success(f"### AI Prediction for {target_col}: **{pred_label}**")
            
            if custom_model['is_binary']:
                prob = pipeline.predict_proba(input_df)[0].max()
                st.info(f"Confidence Score: {prob * 100:.1f}%")

elif selected == "Insights":
    st.markdown('<div class="main-title">Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Data-driven recommendations and feature correlations.</div>', unsafe_allow_html=True)
    
    if data_source == "Sample IBM HR Dataset":
        st.markdown("""
        <div class="glass-card">
        ### 1. Overtime is a Major Flight Risk
        - **Insight**: Employees who work overtime have a significantly higher attrition rate.
        - **Recommendation**: Review workload distribution.
        
        ### 2. Income Disparity and Turnover
        - **Insight**: Employees with lower monthly income show much higher tendencies to leave.
        - **Recommendation**: Conduct a market salary review for entry-level roles.
        
        ### 3. Departmental Stress
        - **Insight**: The Sales department exhibits the highest attrition rates.
        - **Recommendation**: Investigate the high-pressure environment in Sales.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Dynamic Insights for Custom Dataset
        if not st.session_state.get('custom_model_data'):
            st.warning("⚠️ Please train a custom model in the **ML Models** tab first to generate algorithmic insights.")
            st.stop()
            
        target_col = st.session_state['custom_target_col']
        custom_model = st.session_state['custom_model_data']
        st.markdown(f"### Algorithmic Insights against '{target_col}'")
        
        # Calculate Correlations
        df_temp = df.copy()
        if not pd.api.types.is_numeric_dtype(df_temp[target_col]):
            df_temp[target_col] = custom_model['target_encoder'].transform(df_temp[target_col].astype(str))
            
        corr = df_temp.select_dtypes(include=[np.number]).corr()[target_col].sort_values()
        corr = corr.drop(target_col)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top Positive Correlations")
            st.markdown("When these increase, the target variable tends to increase.")
            st.dataframe(corr.tail(5)[::-1])
            
        with col2:
            st.markdown("#### Top Negative Correlations")
            st.markdown("When these increase, the target variable tends to decrease.")
            st.dataframe(corr.head(5))

# --- FOOTER ---
st.markdown("""
<div class="custom-footer">
    Powered by AttritionIQ Auto-ML Engine | Secure Data Analytics Dashboard
</div>
""", unsafe_allow_html=True)
