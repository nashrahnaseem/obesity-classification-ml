"""
Obesity Level Classification - Multi-Model Machine Learning Project
====================================================================

Dataset: ObesityDataSet_raw_and_data_sinthetic.csv
Target: NObeyesdad (7 obesity levels)

Models Implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Evaluation Metrics (6 per model):
- Accuracy, AUC Score, Precision, Recall, F1 Score, MCC Score
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import os
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️  XGBoost not available: {type(e).__name__}")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class ObesityClassificationProject:
    """
    Complete obesity classification project with 6 models and 6 metrics each.
    """
    
    def __init__(self, data_path='ObesityDataSet_raw_and_data_sinthetic.csv'):
        """Initialize the project."""
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.feature_names = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the obesity dataset."""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("=" * 80)
        
        # Load data
        print(f"\n📁 Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"   Dataset shape: {df.shape}")
        print(f"   Features: {df.shape[1] - 1}, Samples: {df.shape[0]}")
        
        # Display info
        print("\n📊 Dataset Overview:")
        print(df.head(3))
        
        print(f"\n🎯 Target Variable: NObeyesdad")
        print("   Class distribution:")
        class_counts = df['NObeyesdad'].value_counts()
        for cls, count in class_counts.items():
            print(f"   - {cls}: {count} samples ({count/len(df)*100:.1f}%)")
        
        print(f"\n🔍 Missing values: {df.isnull().sum().sum()}")
        
        # Encode categorical variables
        print("\n⚙️  Encoding categorical variables...")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('NObeyesdad')  # Remove target
        
        print(f"   Categorical features: {categorical_cols}")
        
        df_encoded = df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
        
        # Separate features and target
        X = df_encoded.drop('NObeyesdad', axis=1)
        y = df_encoded['NObeyesdad']
        
        self.feature_names = X.columns.tolist()
        print(f"\n   Total features after encoding: {len(self.feature_names)}")
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"   Target classes: {len(self.label_encoder.classes_)}")
        print(f"   Class labels: {list(self.label_encoder.classes_)}")
        
        # Split data
        print("\n✂️  Splitting data (80% train, 20% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        
        # Scale features
        print("\n📏 Scaling features with StandardScaler...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✅ Data preprocessing completed!\n")
        
    def initialize_models(self):
        """Initialize all 6 classification models."""
        print("=" * 80)
        print("STEP 2: MODEL INITIALIZATION")
        print("=" * 80)
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                multi_class='multinomial',
                solver='lbfgs'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=15,
                min_samples_split=10
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='euclidean'
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=20,
                n_jobs=-1
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=8,
                learning_rate=0.1,
                eval_metric='mlogloss'
            )
        
        print(f"\n✅ Initialized {len(self.models)} models:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"   {i}. {model_name}")
        
        if not XGBOOST_AVAILABLE:
            print("\n⚠️  Note: XGBoost skipped (install: brew install libomp && pip3 install xgboost)")
        print()
        
    def train_and_evaluate_models(self):
        """Train all models and calculate evaluation metrics."""
        print("=" * 80)
        print("STEP 3: MODEL TRAINING AND EVALUATION")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            print(f"\n{'=' * 80}")
            print(f"🤖 Training: {model_name}")
            print(f"{'=' * 80}")
            
            # Train
            print("   Training in progress...")
            model.fit(self.X_train, self.y_train)
            print("   ✅ Training completed!")
            
            # Predictions
            print("   Making predictions...")
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Probability predictions for AUC
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
            else:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                self.y_test, 
                y_pred_test, 
                y_pred_proba
            )
            
            # Add training accuracy
            metrics['Training Accuracy'] = accuracy_score(self.y_train, y_pred_train)
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred_test
            }
            
            # Display results
            self.display_model_results(model_name, metrics)
            
            # Save model
            self.save_model(model, model_name)
            
        print(f"\n{'=' * 80}")
        print("✅ All models trained and evaluated!")
        print(f"{'=' * 80}\n")
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all 6 evaluation metrics."""
        metrics = {}
        
        # 1. Accuracy
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        
        # 2. AUC Score (multi-class: ovr strategy)
        if y_pred_proba is not None:
            try:
                metrics['AUC Score'] = roc_auc_score(
                    y_true, 
                    y_pred_proba, 
                    multi_class='ovr',
                    average='weighted'
                )
            except Exception:
                metrics['AUC Score'] = np.nan
        else:
            metrics['AUC Score'] = np.nan
        
        # 3. Precision (weighted average)
        metrics['Precision'] = precision_score(
            y_true, 
            y_pred, 
            average='weighted',
            zero_division=0
        )
        
        # 4. Recall (weighted average)
        metrics['Recall'] = recall_score(
            y_true, 
            y_pred, 
            average='weighted',
            zero_division=0
        )
        
        # 5. F1 Score (weighted average)
        metrics['F1 Score'] = f1_score(
            y_true, 
            y_pred, 
            average='weighted',
            zero_division=0
        )
        
        # 6. Matthews Correlation Coefficient
        metrics['MCC Score'] = matthews_corrcoef(y_true, y_pred)
        
        return metrics
    
    def display_model_results(self, model_name, metrics):
        """Display formatted results."""
        print(f"\n   📊 Evaluation Metrics:")
        print(f"   {'-' * 60}")
        print(f"   {'Metric':<25} {'Score':>15}")
        print(f"   {'-' * 60}")
        
        metric_order = [
            'Training Accuracy',
            'Accuracy',
            'AUC Score',
            'Precision',
            'Recall',
            'F1 Score',
            'MCC Score'
        ]
        
        for metric in metric_order:
            if metric in metrics:
                value = metrics[metric]
                if np.isnan(value):
                    print(f"   {metric:<25} {'N/A':>15}")
                else:
                    print(f"   {metric:<25} {value:>15.4f}")
        
        print(f"   {'-' * 60}")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "=" * 80)
        print("STEP 4: MODEL COMPARISON REPORT")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {'ML Model Name': model_name}
            row['Accuracy'] = metrics.get('Accuracy', np.nan)
            row['AUC'] = metrics.get('AUC Score', np.nan)
            row['Precision'] = metrics.get('Precision', np.nan)
            row['Recall'] = metrics.get('Recall', np.nan)
            row['F1'] = metrics.get('F1 Score', np.nan)
            row['MCC'] = metrics.get('MCC Score', np.nan)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1', ascending=False)
        
        print("\n📈 Complete Model Comparison Table:")
        print("=" * 80)
        
        # Display comparison table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', lambda x: f'{x:.4f}')
        print(comparison_df.to_string(index=False))
        
        # Best models by metric
        print("\n🏆 Best Models by Metric:")
        print("=" * 80)
        
        metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        for metric in metrics_list:
            if metric in comparison_df.columns:
                valid_data = comparison_df[comparison_df[metric].notna()]
                if not valid_data.empty:
                    best_idx = valid_data[metric].idxmax()
                    best_model = valid_data.loc[best_idx, 'ML Model Name']
                    best_score = valid_data.loc[best_idx, metric]
                    print(f"   {metric:<15} : {best_model:<25} ({best_score:.4f})")
        
        # Overall best model
        best_overall = comparison_df.iloc[0]
        print("\n" + "=" * 80)
        print(f"🏅 OVERALL BEST MODEL: {best_overall['ML Model Name']}")
        print(f"   Accuracy: {best_overall['Accuracy']:.4f}")
        print(f"   F1 Score: {best_overall['F1']:.4f}")
        print(f"   AUC Score: {best_overall['AUC']:.4f}")
        print("=" * 80)
        
        return comparison_df
    
    def visualize_results(self, comparison_df):
        """Create visualizations."""
        print("\n" + "=" * 80)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Obesity Classification - Model Comparison', 
                     fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        for idx, (ax, metric) in enumerate(zip(axes.flatten(), metrics)):
            if metric in comparison_df.columns:
                data = comparison_df[['ML Model Name', metric]].copy()
                data = data[data[metric].notna()].sort_values(metric, ascending=True)
                
                bars = ax.barh(data['ML Model Name'], data[metric], 
                              color=sns.color_palette("husl", len(data)))
                
                ax.set_xlabel(metric, fontweight='bold')
                ax.set_title(f'{metric} Comparison', fontweight='bold', pad=10)
                ax.set_xlim(0, 1.0)
                
                # Add value labels
                for bar, value in zip(bars, data[metric]):
                    ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{value:.3f}', va='center', fontsize=9)
                
                ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved: model_comparison.png")
        plt.close()
        print("=" * 80)
    
    def save_model(self, model, model_name):
        """Save trained model to file."""
        if not os.path.exists('model'):
            os.makedirs('model')
        
        filename = f"model/{model_name.replace(' ', '_').lower()}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    
    def save_results(self, comparison_df):
        """Save results to CSV."""
        print("\n" + "=" * 80)
        print("STEP 6: SAVING RESULTS")
        print("=" * 80)
        
        output_file = 'classification_results.csv'
        comparison_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        # Save scaler and label encoder
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('model/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ Preprocessors saved to model/ directory")
        print("=" * 80)
    
    def run_complete_pipeline(self):
        """Run the complete pipeline."""
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  OBESITY LEVEL CLASSIFICATION PROJECT".center(78) + "║")
        print("║" + "  Multi-Model Machine Learning Analysis".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        
        self.load_and_preprocess_data()
        self.initialize_models()
        self.train_and_evaluate_models()
        comparison_df = self.generate_comparison_report()
        self.visualize_results(comparison_df)
        self.save_results(comparison_df)
        
        print("\n" + "=" * 80)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nGenerated Files:")
        print("   1. classification_results.csv - Metrics comparison table")
        print("   2. model_comparison.png - Visual comparison charts")
        print("   3. model/ - Directory with all 6 saved models")
        print("\n" + "=" * 80 + "\n")


def main():
    """Main execution function."""
    project = ObesityClassificationProject(
        data_path='ObesityDataSet_raw_and_data_sinthetic.csv'
    )
    project.run_complete_pipeline()


if __name__ == "__main__":
    main()
