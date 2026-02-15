"""
Streamlit App for Obesity Level Classification
===============================================
Interactive web interface for multi-model classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Obesity Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the obesity dataset."""
    try:
        df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure ObesityDataSet_raw_and_data_sinthetic.csv is in the project directory.")
        return None


@st.cache_data
def load_results():
    """Load classification results."""
    try:
        results_df = pd.read_csv('classification_results.csv')
        return results_df
    except FileNotFoundError:
        st.warning("⚠️ Results file not found. Please run obesity_classification.py first.")
        return None


@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    model_dir = 'model'
    
    if not os.path.exists(model_dir):
        return None
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'k-nearest_neighbors.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    # Load preprocessors
    try:
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        models['_scaler'] = scaler
        models['_label_encoder'] = label_encoder
    except:
        pass
    
    return models if models else None


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Obesity Level Classification</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Multi-Model Machine Learning Project")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "📊 Dataset Overview", "🤖 Model Comparison", 
         "🎯 Make Prediction", "📈 Analysis"]
    )
    
    # Load data
    df = load_data()
    results_df = load_results()
    models = load_models()
    
    # Page routing
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Dataset Overview":
        show_dataset_overview(df)
    elif page == "🤖 Model Comparison":
        show_model_comparison(results_df)
    elif page == "🎯 Make Prediction":
        show_prediction(models, df)
    elif page == "📈 Analysis":
        show_analysis(df, results_df)


def show_home():
    """Display home page."""
    st.markdown("## 🎯 Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 📁 Dataset\n**Obesity Level Dataset**\n- 2,111 samples\n- 17 features\n- 7 classes")
    
    with col2:
        st.success("### 🤖 Models\n**6 ML Algorithms**\n- Logistic Regression\n- Decision Tree\n- KNN\n- Naive Bayes\n- Random Forest\n- XGBoost")
    
    with col3:
        st.warning("### 📊 Metrics\n**6 Evaluation Metrics**\n- Accuracy\n- AUC Score\n- Precision\n- Recall\n- F1 Score\n- MCC Score")
    
    st.markdown("---")
    
    st.markdown("## 📖 About This Project")
    st.write("""
    This project implements a comprehensive obesity level classification system using multiple machine learning algorithms.
    The goal is to predict obesity levels based on various lifestyle and physical attributes.
    
    **Key Features:**
    - ✅ Complete data preprocessing pipeline
    - ✅ Implementation of 6 classification algorithms
    - ✅ Comprehensive evaluation with 6 metrics per model
    - ✅ Interactive Streamlit web interface
    - ✅ Model comparison and visualization
    - ✅ Real-time prediction capability
    """)
    
    st.markdown("## 🎓 Obesity Levels")
    
    levels_df = pd.DataFrame({
        'Level': ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 
                  'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III'],
        'Description': ['Below healthy weight', 'Healthy weight range', 'Slightly overweight',
                       'Moderately overweight', 'Obese class I', 'Obese class II', 'Obese class III']
    })
    
    st.dataframe(levels_df, use_container_width=True)


def show_dataset_overview(df):
    """Display dataset overview."""
    st.markdown("## 📊 Dataset Overview")
    
    if df is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Total Features", df.shape[1] - 1)
    with col3:
        st.metric("Classes", df['NObeyesdad'].nunique())
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    # Show sample data
    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Class distribution
    st.markdown("### 📈 Class Distribution")
    class_counts = df['NObeyesdad'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(class_counts)))
    ax.set_title('Distribution of Obesity Levels', fontsize=14, fontweight='bold')
    ax.set_xlabel('Obesity Level', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature description
    st.markdown("### 📝 Feature Descriptions")
    
    features_info = {
        'Gender': 'Male or Female',
        'Age': 'Age in years',
        'Height': 'Height in meters',
        'Weight': 'Weight in kilograms',
        'family_history_with_overweight': 'Family history of overweight (yes/no)',
        'FAVC': 'Frequent consumption of high caloric food (yes/no)',
        'FCVC': 'Frequency of consumption of vegetables (1-3)',
        'NCP': 'Number of main meals (1-4)',
        'CAEC': 'Consumption of food between meals',
        'SMOKE': 'Smoking (yes/no)',
        'CH2O': 'Daily water consumption (1-3)',
        'SCC': 'Calorie consumption monitoring (yes/no)',
        'FAF': 'Physical activity frequency (0-3)',
        'TUE': 'Time using technology devices (0-2)',
        'CALC': 'Alcohol consumption',
        'MTRANS': 'Transportation used',
        'NObeyesdad': 'Obesity level (Target variable)'
    }
    
    features_df = pd.DataFrame(list(features_info.items()), 
                               columns=['Feature', 'Description'])
    st.dataframe(features_df, use_container_width=True, hide_index=True)


def show_model_comparison(results_df):
    """Display model comparison."""
    st.markdown("## 🤖 Model Comparison")
    
    if results_df is None:
        st.error("❌ Results not available. Please run obesity_classification.py first.")
        st.code("python3 obesity_classification.py", language="bash")
        return
    
    # Display results table
    st.markdown("### 📊 Complete Metrics Comparison")
    
    # Format and display
    styled_df = results_df.style.format({
        'Accuracy': '{:.4f}',
        'AUC': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1': '{:.4f}',
        'MCC': '{:.4f}'
    }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Best model highlight
    st.markdown("---")
    best_model_idx = results_df['F1'].idxmax()
    best_model = results_df.loc[best_model_idx]
    
    st.markdown("### 🏆 Best Performing Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"### {best_model['ML Model Name']}")
    with col2:
        st.metric("F1 Score", f"{best_model['F1']:.4f}")
    with col3:
        st.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
    
    # Visualization
    st.markdown("---")
    st.markdown("### 📈 Visual Comparison")
    
    # Check if image exists
    if os.path.exists('model_comparison.png'):
        st.image('model_comparison.png', use_column_width=True)
    else:
        st.warning("Visualization not found. Run obesity_classification.py to generate charts.")
    
    # Metric-wise comparison
    st.markdown("---")
    st.markdown("### 🎯 Metric-wise Best Models")
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    cols = st.columns(3)
    for i, metric in enumerate(metrics):
        with cols[i % 3]:
            best_idx = results_df[metric].idxmax()
            best_value = results_df.loc[best_idx, metric]
            best_name = results_df.loc[best_idx, 'ML Model Name']
            st.info(f"**{metric}**\n\n{best_name}\n\n{best_value:.4f}")


def show_prediction(models, df):
    """Display prediction interface."""
    st.markdown("## 🎯 Make a Prediction")
    
    if models is None:
        st.error("❌ Models not loaded. Please run obesity_classification.py first.")
        return
    
    if df is None:
        st.error("❌ Dataset not loaded.")
        return
    
    st.markdown("### Enter Patient Information")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        age = st.slider("Age", 10, 70, 25)
        height = st.slider("Height (m)", 1.4, 2.0, 1.7, 0.01)
        weight = st.slider("Weight (kg)", 40, 150, 70)
        family_history = st.selectbox("Family History with Overweight", ['yes', 'no'])
        favc = st.selectbox("Frequent High Caloric Food", ['yes', 'no'])
        fcvc = st.slider("Vegetable Consumption Frequency", 1, 3, 2)
        ncp = st.slider("Number of Main Meals", 1, 4, 3)
    
    with col2:
        caec = st.selectbox("Food Between Meals", ['no', 'Sometimes', 'Frequently', 'Always'])
        smoke = st.selectbox("Smoking", ['yes', 'no'])
        ch2o = st.slider("Daily Water Consumption", 1, 3, 2)
        scc = st.selectbox("Calorie Monitoring", ['yes', 'no'])
        faf = st.slider("Physical Activity Frequency", 0, 3, 1)
        tue = st.slider("Technology Use Time", 0, 2, 1)
        calc = st.selectbox("Alcohol Consumption", ['no', 'Sometimes', 'Frequently', 'Always'])
        mtrans = st.selectbox("Transportation", ['Walking', 'Public_Transportation', 'Automobile', 'Motorbike', 'Bike'])
    
    if st.button("🔮 Predict Obesity Level", type="primary"):
        # Prepare input
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'family_history_with_overweight': [family_history],
            'FAVC': [favc],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CAEC': [caec],
            'SMOKE': [smoke],
            'CH2O': [ch2o],
            'SCC': [scc],
            'FAF': [faf],
            'TUE': [tue],
            'CALC': [calc],
            'MTRANS': [mtrans]
        })
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 
                           'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on original data to get all categories
            le.fit(df[col])
            input_data[col] = le.transform(input_data[col])
        
        # Scale
        if '_scaler' in models:
            input_scaled = models['_scaler'].transform(input_data)
        else:
            input_scaled = input_data.values
        
        # Make predictions with all models
        st.markdown("---")
        st.markdown("### 🎯 Predictions from All Models")
        
        predictions = {}
        for model_name, model in models.items():
            if not model_name.startswith('_'):
                try:
                    pred = model.predict(input_scaled)[0]
                    if '_label_encoder' in models:
                        pred_label = models['_label_encoder'].inverse_transform([pred])[0]
                    else:
                        pred_label = pred
                    predictions[model_name] = pred_label
                    
                    # Get probability if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_scaled)[0]
                        confidence = proba.max()
                    else:
                        confidence = None
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.success(f"**{model_name}**: {pred_label}")
                    with col2:
                        if confidence:
                            st.metric("Confidence", f"{confidence:.1%}")
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
        
        # Consensus prediction
        if predictions:
            from collections import Counter
            most_common = Counter(predictions.values()).most_common(1)[0]
            
            st.markdown("---")
            st.markdown("### 🎖️ Consensus Prediction")
            st.success(f"# {most_common[0]}")
            st.info(f"**Agreement**: {most_common[1]}/{len(predictions)} models")


def show_analysis(df, results_df):
    """Display additional analysis."""
    st.markdown("## 📈 Detailed Analysis")
    
    if df is None or results_df is None:
        st.error("❌ Data not available.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📊 Feature Analysis", "🎯 Model Insights", "📉 Correlations"])
    
    with tab1:
        st.markdown("### Feature Distributions")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select Feature", numeric_cols)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        df[selected_feature].hist(bins=30, ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title(f'Distribution of {selected_feature}')
        ax1.set_xlabel(selected_feature)
        ax1.set_ylabel('Frequency')
        
        # Box plot by obesity level
        df.boxplot(column=selected_feature, by='NObeyesdad', ax=ax2, rot=45)
        ax2.set_title(f'{selected_feature} by Obesity Level')
        plt.suptitle('')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.markdown("### Model Performance Insights")
        
        if results_df is not None:
            # Performance comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(results_df))
            width = 0.15
            
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
                ax.bar(x + i*width, results_df[metric], width, label=metric, color=color)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(results_df['ML Model Name'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab3:
        st.markdown("### Feature Correlations")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
