import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate():
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv('application_record.csv')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Target variable mapping (0 = Approved, 1 = Rejected)
    # 2. Preprocessing & Feature Engineering
    le_emp = LabelEncoder()
    df['Employment'] = le_emp.fit_transform(df['Employment'])
    
    le_mar = LabelEncoder()
    df['Married'] = le_mar.fit_transform(df['Married'])
    
    X = df.drop('Approved', axis=1)
    y = df['Approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Building
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_f1 = -1
    best_model_name = ""
    
    results = []
    
    plt.rcParams["figure.figsize"] = (8, 6)
    sns.set_theme(style="whitegrid", context="talk")
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc
        })
        
        # Save confusion matrix for this model
        plt.figure()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Approved (0)', 'Rejected (1)'],
                    yticklabels=['Approved (0)', 'Rejected (1)'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'static/images/cm_{name.replace(" ", "_").lower()}.png', bbox_inches='tight', transparent=True)
        plt.close()
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    print(f"Best Model Selected: {best_model_name} with F1 {best_f1:.4f}")
    
    # Plot performance comparison
    results_df = pd.DataFrame(results)
    print("Performance Summary:")
    print(results_df)
    results_df.to_csv('models/model_metrics.csv', index=False)
    
    # Feature Importance (only for Tree based)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure()
        sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
        plt.title('Feature Importance')
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')
        plt.savefig('static/images/feature_importance.png', bbox_inches='tight', transparent=True)
        plt.close()
    
    # 4. Save best model and scalers
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_emp, 'models/le_emp.pkl')
    joblib.dump(le_mar, 'models/le_mar.pkl')
    print("Models and artifacts saved in 'models/' and 'static/images/'")

if __name__ == '__main__':
    train_and_evaluate()
