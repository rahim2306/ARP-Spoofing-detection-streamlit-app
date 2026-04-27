import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report
)
import xgboost as xgb
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: #1f2937;
    }
    .attack-badge {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .normal-badge {
        background: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .prediction-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #1f2937;
    }
    .prediction-box h4 {
        color: #111827;
        margin-top: 0;
    }
    .prediction-box p {
        color: #4b5563;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING (Cached)
# ============================================================================
@st.cache_resource
def load_models():
    """Load all models and preprocessing artifacts"""
    models_dir = Path("Models")
    
    artifacts = {}
    
    try:
        # Load XGBoost model
        artifacts['xgb_model'] = xgb.Booster()
        artifacts['xgb_model'].load_model(str(models_dir / 'XGBoost' / 'xgboost_model.json'))
        
        # Load LSTM model
        artifacts['lstm_model'] = tf.keras.models.load_model(
            str(models_dir / 'RNN' / 'lstm_model.keras'),
            compile=False
        )
        
        # Load preprocessing artifacts from RNN folder using joblib (matching training notebook)
        artifacts['label_encoder'] = joblib.load(models_dir / 'RNN' / 'le.pkl')
        artifacts['variance_selector'] = joblib.load(models_dir / 'RNN' / 'variance_selector.pkl')
        artifacts['scaler'] = joblib.load(models_dir / 'RNN' / 'scaler.pkl')
        
        # Store feature names from variance selector
        artifacts['selected_features_mask'] = artifacts['variance_selector'].get_support()
        
        return artifacts, None
    
    except Exception as e:
        return None, f"Error loading models: {str(e)}"

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================
def preprocess_data(df, artifacts):
    """
    Preprocess input data exactly as done during training.
    Returns (X_train_processed, y_true, error_message)
    """
    try:
        # Make a copy
        df_processed = df.copy()
        
        # Check if ground truth is available
        has_ground_truth = 'Sub_Cat' in df_processed.columns
        
        # Store original indices
        original_indices = df_processed.index.tolist()
        
        # 1. Filter to 2 classes if Sub_Cat present
        if has_ground_truth:
            df_processed = df_processed[
                df_processed['Sub_Cat'].isin(['MITM ARP Spoofing', 'Normal'])
            ]
            if len(df_processed) == 0:
                return None, None, "No valid classes found after filtering"
        
        # 2. Clean
        df_processed.drop_duplicates(inplace=True)
        df_processed.dropna(inplace=True)
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_processed.dropna(inplace=True)
        
        if len(df_processed) == 0:
            return None, None, "No data remaining after cleaning"
        
        # 3. Drop identifier columns
        drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
        df_processed.drop(
            columns=[c for c in drop_cols if c in df_processed.columns], 
            inplace=True
        )
        
        # 4. Extract ground truth if available
        y_true = None
        if has_ground_truth:
            y_true = df_processed['Sub_Cat'].values
            # Check if Label column exists for alternative ground truth
            if 'Label' in df_processed.columns:
                # Use Label if Sub_Cat not sufficient
                pass
        
        # 5. Drop target and non-numeric columns
        X = df_processed.drop(
            columns=['Label', 'Cat', 'Sub_Cat'], 
            errors='ignore'
        )
        X.drop(columns=X.select_dtypes(exclude=['number']).columns, inplace=True)
        
        if X.shape[1] == 0:
            return None, None, "No numeric features remaining after preprocessing"
        
        # 6. Apply variance selector
        selector = artifacts['variance_selector']
        X_selected = pd.DataFrame(
            selector.transform(X),
            columns=X.columns[artifacts['selected_features_mask']]
        )
        
        return X_selected, y_true, None
    
    except Exception as e:
        return None, None, f"Preprocessing error: {str(e)}"

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def predict_xgboost(X, artifacts):
    """Make predictions using XGBoost model"""
    try:
        xgb_model = artifacts['xgb_model']
        dmatrix = xgb.DMatrix(X)
        
        y_proba = xgb_model.predict(dmatrix)
        # XGBoost outputs probability of class 1
        y_pred = (y_proba > 0.5).astype(int)
        
        return y_pred, y_proba, None
    except Exception as e:
        return None, None, f"XGBoost prediction error: {str(e)}"

def predict_lstm(X, artifacts):
    """Make predictions using LSTM model"""
    try:
        lstm_model = artifacts['lstm_model']
        scaler = artifacts['scaler']
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Reshape to 3D (samples, timesteps, features)
        X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Predict
        y_proba = lstm_model.predict(X_lstm, verbose=0).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        
        return y_pred, y_proba, None
    except Exception as e:
        return None, None, f"LSTM prediction error: {str(e)}"

def decode_predictions(y_pred, artifacts):
    """Convert numeric predictions to labels"""
    le = artifacts['label_encoder']
    return le.inverse_transform(y_pred)

# ============================================================================
# UI COMPONENTS
# ============================================================================
def display_summary_stats(total_rows, attack_count, normal_count, attack_percentage):
    """Display summary statistics in metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#6b7280;">Total Rows</h3>
                <h2 style="margin:0.5rem 0; color:#1f2937;">{total_rows:,}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#6b7280;">Attack Count</h3>
                <h2 style="margin:0.5rem 0; color:#991b1b;">{attack_count:,}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#6b7280;">Normal Count</h3>
                <h2 style="margin:0.5rem 0; color:#065f46;">{normal_count:,}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#6b7280;">Attack %</h3>
                <h2 style="margin:0.5rem 0; color:#d97706;">{attack_percentage:.1f}%</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )

def display_confusion_matrix(y_true, y_pred_labels, model_name, artifacts):
    """Display confusion matrix as plotly heatmap"""
    le = artifacts['label_encoder']
    cm = confusion_matrix(y_true, y_pred_labels, labels=le.classes_)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=le.classes_,
        y=le.classes_,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title=f"{model_name} - Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_feature_importance(artifacts, feature_names):
    """Display XGBoost feature importance"""
    try:
        xgb_model = artifacts['xgb_model']
        importance = xgb_model.get_score(importance_type='gain')
        
        # Get top 15 features
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Feature', 'Importance']
        ).nlargest(15, 'Importance')
        
        # Map feature names
        importance_df['Feature'] = importance_df['Feature'].map(
            lambda x: feature_names[int(x.replace('f', ''))] if x.startswith('f') else x
        )
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Feature Importance (XGBoost)',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")

def display_results_table(X_processed, y_pred_labels, y_proba, original_df, key):
    """Display prediction results in a styled table"""
    # Calculate confidence based on the predicted class certainty
    # In binary: prob for 'Normal' (1), so (1 - prob) is certainty for 'MITM' (0)
    confidences = []
    for i, label in enumerate(y_pred_labels):
        prob = y_proba[i]
        certainty = prob if label == 'Normal' else 1.0 - prob
        confidences.append(f"{certainty:.2%}")

    # Create results dataframe
    results_df = pd.DataFrame({
        'Row': range(1, len(y_pred_labels) + 1),
        'Prediction': y_pred_labels,
        'Confidence': confidences
    })
    
    # Style the dataframe
    def color_predictions(val):
        if 'MITM' in str(val):
            return 'background-color: #fee2e2; color: #991b1b'
        elif 'Normal' in str(val):
            return 'background-color: #d1fae5; color: #065f46'
        return ''
    
    # Update applymap to map for Pandas 2.x support
    styled_df = results_df.style.map(
        color_predictions, 
        subset=['Prediction']
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )
    
    # Rename key and filename to ensure uniqueness and freshness
    csv = results_df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download {key.upper()} Results.csv",
        data=csv,
        file_name=f"nids_{key}_report.csv",
        mime="text/csv",
        key=f"dl_btn_v3_{key}"
    )

def display_classification_report(y_true, y_pred_labels, model_name, artifacts):
    """Display classification report"""
    le = artifacts['label_encoder']
    report = classification_report(
        y_true, 
        y_pred_labels,
        labels=le.classes_,
        target_names=le.classes_,
        output_dict=True
    )
    
    report_df = pd.DataFrame(report).transpose()
    
    st.markdown(f"### {model_name} - Classification Report")
    
    # Format the report
    formatted_report = report_df.copy()
    for col in formatted_report.columns:
        if col in ['precision', 'recall', 'f1-score']:
            formatted_report[col] = formatted_report[col].apply(
                lambda x: f"{x:.3f}" if pd.notnull(x) else ""
            )
        elif col == 'support':
            formatted_report[col] = formatted_report[col].apply(
                lambda x: f"{int(x)}" if pd.notnull(x) else ""
            )
    
    st.dataframe(formatted_report, use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Network Intrusion Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">MITM ARP Spoofing Detection using Machine Learning & Deep Learning</p>', unsafe_allow_html=True)
    
    # Load models
    artifacts, error = load_models()
    
    if error:
        st.error(f"❌ Failed to load models: {error}")
        st.stop()
    
    st.success("✅ Models loaded successfully! Ready for prediction.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Selection")
        model_choice = st.radio(
            "Choose Model for Prediction",
            ["Both (Side-by-Side)", "XGBoost Only", "LSTM Only"],
            help="Select which model(s) to use for predictions"
        )
        
        st.markdown("---")
        st.markdown("## 📈 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "XGBoost",
                "99.0%",
                "Accuracy",
                help="Slight overfitting after round 150"
            )
        with col2:
            st.metric(
                "LSTM",
                "97.3%",
                "Validation Acc.",
                help="Healthy convergence, no overfitting"
            )
        
        st.markdown("---")
        st.markdown("## 📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload IoTID20 format CSV file for prediction"
        )
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.info(
            """
            This system detects MITM ARP Spoofing attacks in 
            network traffic using two state-of-the-art models:
            
            • **XGBoost**: Gradient boosting classifier
            • **LSTM**: Deep learning sequential model
            
            Both models were trained on the IoTID20 dataset
            for binary classification.
            """
        )
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                X_processed, y_true, prep_error = preprocess_data(df, artifacts)
            
            if prep_error:
                st.error(f"❌ {prep_error}")
                st.stop()
            
            st.success(f"✅ Data preprocessed successfully. {X_processed.shape[0]} rows, {X_processed.shape[1]} features")
            
            # Make predictions based on model choice
            has_ground_truth = y_true is not None
            
            if model_choice in ["XGBoost Only", "Both (Side-by-Side)"]:
                st.markdown("---")
                st.markdown("## 🔮 XGBoost Predictions")
                
                with st.spinner("Running XGBoost predictions..."):
                    y_pred_xgb, y_proba_xgb, xgb_error = predict_xgboost(X_processed, artifacts)
                
                if xgb_error:
                    st.error(f"❌ {xgb_error}")
                else:
                    y_pred_labels_xgb = decode_predictions(y_pred_xgb, artifacts)
                    
                    # Summary stats
                    attack_count = np.sum(y_pred_labels_xgb == 'MITM ARP Spoofing')
                    normal_count = np.sum(y_pred_labels_xgb == 'Normal')
                    total = len(y_pred_labels_xgb)
                    attack_percentage = (attack_count / total) * 100
                    
                    display_summary_stats(total, attack_count, normal_count, attack_percentage)
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    display_results_table(X_processed, y_pred_labels_xgb, y_proba_xgb, df, "xgb")
                    
                    # Display metrics if ground truth available
                    if has_ground_truth:
                        accuracy = accuracy_score(y_true, y_pred_labels_xgb)
                        st.metric("XGBoost Accuracy", f"{accuracy:.2%}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            display_confusion_matrix(y_true, y_pred_labels_xgb, "XGBoost", artifacts)
                        with col2:
                            display_classification_report(y_true, y_pred_labels_xgb, "XGBoost", artifacts)
                    
                    # Feature importance
                    st.markdown("### Feature Importance Analysis")
                    display_feature_importance(artifacts, X_processed.columns)
            
            if model_choice in ["LSTM Only", "Both (Side-by-Side)"]:
                st.markdown("---")
                st.markdown("## 🧠 LSTM Predictions")
                
                with st.spinner("Running LSTM predictions..."):
                    y_pred_lstm, y_proba_lstm, lstm_error = predict_lstm(X_processed, artifacts)
                
                if lstm_error:
                    st.error(f"❌ {lstm_error}")
                else:
                    y_pred_labels_lstm = decode_predictions(y_pred_lstm, artifacts)
                    
                    # Summary stats
                    attack_count = np.sum(y_pred_labels_lstm == 'MITM ARP Spoofing')
                    normal_count = np.sum(y_pred_labels_lstm == 'Normal')
                    total = len(y_pred_labels_lstm)
                    attack_percentage = (attack_count / total) * 100
                    
                    display_summary_stats(total, attack_count, normal_count, attack_percentage)
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    display_results_table(X_processed, y_pred_labels_lstm, y_proba_lstm, df, "lstm")
                    
                    # Display metrics if ground truth available
                    if has_ground_truth:
                        accuracy = accuracy_score(y_true, y_pred_labels_lstm)
                        st.metric("LSTM Accuracy", f"{accuracy:.2%}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            display_confusion_matrix(y_true, y_pred_labels_lstm, "LSTM", artifacts)
                        with col2:
                            display_classification_report(y_true, y_pred_labels_lstm, "LSTM", artifacts)
            
            # Model comparison if both selected and ground truth available
            if model_choice == "Both (Side-by-Side)" and has_ground_truth and not xgb_error and not lstm_error:
                st.markdown("---")
                st.markdown("## 📊 Model Comparison")
                
                comparison_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Attack Detected', 'Normal Detected'],
                    'XGBoost': [
                        f"{accuracy_score(y_true, y_pred_labels_xgb):.2%}",
                        f"{np.sum((y_true == 'MITM ARP Spoofing') & (y_pred_labels_xgb == 'MITM ARP Spoofing'))}",
                        f"{np.sum((y_true == 'Normal') & (y_pred_labels_xgb == 'Normal'))}"
                    ],
                    'LSTM': [
                        f"{accuracy_score(y_true, y_pred_labels_lstm):.2%}",
                        f"{np.sum((y_true == 'MITM ARP Spoofing') & (y_pred_labels_lstm == 'MITM ARP Spoofing'))}",
                        f"{np.sum((y_true == 'Normal') & (y_pred_labels_lstm == 'Normal'))}"
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Agreement analysis
                agreement = np.mean(y_pred_labels_xgb == y_pred_labels_lstm) * 100
                st.metric("Model Agreement Rate", f"{agreement:.1f}%")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
    
    else:
        # Display placeholder when no file is uploaded
        st.markdown("---")
        st.markdown("## 🚀 Getting Started")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div class="prediction-box">
                    <h4>📤 1. Upload Data</h4>
                    <p>Upload your CSV file in IoTID20 format using the sidebar</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div class="prediction-box">
                    <h4>🔍 2. Select Model</h4>
                    <p>Choose XGBoost, LSTM, or both for side-by-side comparison</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                """
                <div class="prediction-box">
                    <h4>📊 3. Analyze Results</h4>
                    <p>View predictions, confidence scores, and performance metrics</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.markdown("## 📊 Expected Data Format")
        
        st.info(
            """
            Your CSV file should contain network traffic features in IoTID20 format.
            Expected columns include:
            
            - `Flow ID`, `Src IP`, `Dst IP`, `Timestamp` (identifiers - will be dropped)
            - `Label`, `Cat`, `Sub_Cat` (target columns - will be dropped during preprocessing)
            - Numeric feature columns (various network metrics)
            
            **Optional**: Include `Sub_Cat` column with values 'MITM ARP Spoofing' or 'Normal' 
            for accuracy evaluation and confusion matrix display.
            """
        )

if __name__ == "__main__":
    main()