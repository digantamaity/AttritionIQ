import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Configuration
DATA_DIR = 'data'
CHARTS_DIR = 'charts'
MODELS_DIR = 'models'
DATA_URL = 'https://raw.githubusercontent.com/pavopax/ibm-hr-analytics-attrition-dataset/master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
DATA_FILE = os.path.join(DATA_DIR, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
MODEL_FILE = os.path.join(MODELS_DIR, 'model.pkl')

# Create Directories
for directory in [DATA_DIR, CHARTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# 1. Download Data if not present
if not os.path.exists(DATA_FILE):
    print(f"Downloading dataset to {DATA_FILE}...")
    response = requests.get(DATA_URL)
    with open(DATA_FILE, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

# 2. Load Data
print("Loading data...")
df = pd.read_csv(DATA_FILE)

# 3. Data Cleaning
# Drop useless columns
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 4. EDA & Professional Charts (Dark Theme)
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

print("Generating EDA Charts...")
# Chart 1: Attrition Count
plt.figure(figsize=(8, 5))
sns.countplot(x='Attrition', data=df, palette='viridis')
plt.title('Attrition Count', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'attrition_count.png'), dpi=300)
plt.close()

# Chart 2: Monthly Income vs Attrition
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='magma')
plt.title('Monthly Income vs Attrition', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'monthly_income_attrition.png'), dpi=300)
plt.close()

# Chart 3: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Attrition', multiple='stack', palette='coolwarm', bins=30)
plt.title('Age Distribution by Attrition', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'age_distribution.png'), dpi=300)
plt.close()

# Chart 4: Department vs Attrition
plt.figure(figsize=(10, 6))
sns.countplot(x='Department', hue='Attrition', data=df, palette='Set2')
plt.title('Department vs Attrition', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'department_attrition.png'), dpi=300)
plt.close()

# Chart 5: Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(16, 12))
corr = numeric_df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'correlation_heatmap.png'), dpi=300)
plt.close()

# 5. Preprocessing & Feature Engineering
print("Preprocessing Data...")
# Target Encoding
le = LabelEncoder()
y = le.fit_transform(df['Attrition']) # Yes: 1, No: 0
X = df.drop(columns=['Attrition'])

# Identify numerical and categorical columns
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
    ])

# Fit and transform
X_processed = preprocessor.fit_transform(X)

# Get feature names after encoding for feature importance plot
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
feature_names = num_features + cat_feature_names

# 6. Train Test Split & SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 7. Model Training & Comparison
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = []
best_model = None
best_auc = 0
best_model_name = ""

print("Training Models...")
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'ROC-AUC': auc
    })
    
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Save Model Comparison Chart
plt.figure(figsize=(10, 6))
sns.barplot(x='ROC-AUC', y='Model', data=results_df.sort_values(by='ROC-AUC', ascending=False), palette='mako')
plt.title('Model Comparison by ROC-AUC', fontsize=16)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'model_comparison.png'), dpi=300)
plt.close()

# 8. Save Best Model and Preprocessor
print(f"\nSaving best model ({best_model_name}) to {MODEL_FILE}...")
export_data = {
    'preprocessor': preprocessor,
    'model': best_model,
    'feature_names': feature_names,
    'target_encoder': le,
    'num_features': num_features,
    'cat_features': cat_features,
    'model_name': best_model_name,
    'metrics': results_df[results_df['Model'] == best_model_name].iloc[0].to_dict()
}
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(export_data, f)

print("Training pipeline completed successfully!")
