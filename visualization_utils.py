import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️ seaborn not available - some visualizations may be simplified")

class ChurnVisualizer:
    """
    Creates beautiful visualizations for churn prediction results
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def create_performance_metrics_card(self, metrics: Dict) -> go.Figure:
        """
        Create a beautiful performance metrics card
        """
        # Extract metrics
        accuracy = metrics.get('accuracy', 0)
        f1_score = metrics.get('f1_score', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        # Create gauge charts
        fig = sp.make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall')
        )
        
        # Accuracy gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=accuracy * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Accuracy (%)"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 70], 'color': self.color_palette['danger']},
                        {'range': [70, 85], 'color': self.color_palette['warning']},
                        {'range': [85, 100], 'color': self.color_palette['success']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # F1 Score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=f1_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "F1 Score (%)"},
                delta={'reference': 85},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['secondary']},
                    'steps': [
                        {'range': [0, 70], 'color': self.color_palette['danger']},
                        {'range': [70, 85], 'color': self.color_palette['warning']},
                        {'range': [85, 100], 'color': self.color_palette['success']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ),
            row=1, col=2
        )
        
        # Precision gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=precision * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Precision (%)"},
                delta={'reference': 85},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['info']},
                    'steps': [
                        {'range': [0, 70], 'color': self.color_palette['danger']},
                        {'range': [70, 85], 'color': self.color_palette['warning']},
                        {'range': [85, 100], 'color': self.color_palette['success']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ),
            row=2, col=1
        )
        
        # Recall gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=recall * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Recall (%)"},
                delta={'reference': 85},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['warning']},
                    'steps': [
                        {'range': [0, 70], 'color': self.color_palette['danger']},
                        {'range': [70, 85], 'color': self.color_palette['warning']},
                        {'range': [85, 100], 'color': self.color_palette['success']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Model Performance Metrics",
            title_x=0.5,
            font=dict(size=14)
        )
        
        return fig
    
    def create_churn_distribution_chart(self, df: pd.DataFrame, churn_column: str = 'churn') -> go.Figure:
        """
        Create churn distribution visualization
        """
        churn_counts = df[churn_column].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Loyal Customers', 'Churned Customers'],
                values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                hole=0.4,
                marker_colors=[self.color_palette['success'], self.color_palette['danger']],
                textinfo='label+percent',
                textfont_size=16,
                hoverinfo='label+value'
            )
        ])
        
        fig.update_layout(
            title_text="Customer Churn Distribution",
            title_x=0.5,
            showlegend=True,
            height=500,
            font=dict(size=14)
        )
        
        return fig
    
    def create_churn_probability_distribution(self, df: pd.DataFrame, probability_column: str = 'churn_probability') -> go.Figure:
        """
        Create churn probability distribution histogram
        """
        fig = go.Figure()
        
        # Create histogram for churn probabilities
        fig.add_trace(go.Histogram(
            x=df[probability_column],
            nbinsx=30,
            name='Churn Probability',
            marker_color=self.color_palette['primary'],
            opacity=0.7
        ))
        
        # Add vertical lines for risk thresholds
        fig.add_vline(x=0.3, line_dash="dash", line_color=self.color_palette['success'], 
                     annotation_text="Low Risk Threshold")
        fig.add_vline(x=0.7, line_dash="dash", line_color=self.color_palette['danger'], 
                     annotation_text="High Risk Threshold")
        
        fig.update_layout(
            title_text="Churn Probability Distribution",
            title_x=0.5,
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            height=500,
            font=dict(size=14),
            showlegend=False
        )
        
        return fig
    
    def create_risk_level_chart(self, df: pd.DataFrame, risk_column: str = 'risk_level') -> go.Figure:
        """
        Create risk level breakdown chart
        """
        risk_counts = df[risk_column].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(risk_counts.index),
                y=list(risk_counts.values),
                marker_color=[
                    self.color_palette['success'],
                    self.color_palette['warning'],
                    self.color_palette['danger']
                ],
                text=list(risk_counts.values),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title_text="Customer Risk Level Distribution",
            title_x=0.5,
            xaxis_title="Risk Level",
            yaxis_title="Number of Customers",
            height=500,
            font=dict(size=14),
            showlegend=False
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """
        Create feature importance visualization
        """
        if feature_importance is None or feature_importance.empty:
            return None
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color=self.color_palette['primary'],
                text=[f"{val:.3f}" for val in top_features['importance']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title_text=f"Top {top_n} Most Important Features",
            title_x=0.5,
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            height=max(400, top_n * 30),
            font=dict(size=14),
            showlegend=False
        )
        
        return fig
    
    def create_confusion_matrix(self, y_true: List, y_pred: List) -> go.Figure:
        """
        Create confusion matrix heatmap
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: No Churn', 'Predicted: Churn'],
            y=['Actual: No Churn', 'Actual: Churn'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title_text="Confusion Matrix",
            title_x=0.5,
            height=500,
            font=dict(size=14)
        )
        
        return fig
    
    def create_customer_segments_chart(self, df: pd.DataFrame, 
                                     segment_column: str = 'churn_probability',
                                     n_segments: int = 5) -> go.Figure:
        """
        Create customer segmentation chart
        """
        # Create segments based on churn probability
        df_copy = df.copy()
        df_copy['segment'] = pd.cut(
            df_copy[segment_column], 
            bins=n_segments, 
            labels=[f'Segment {i+1}' for i in range(n_segments)]
        )
        
        segment_stats = df_copy.groupby('segment').agg({
            'churn_probability': ['mean', 'count']
        }).round(3)
        
        segment_stats.columns = ['Avg_Churn_Prob', 'Customer_Count']
        segment_stats = segment_stats.reset_index()
        
        fig = go.Figure()
        
        # Add bar chart for customer count
        fig.add_trace(go.Bar(
            x=segment_stats['segment'],
            y=segment_stats['Customer_Count'],
            name='Customer Count',
            yaxis='y',
            marker_color=self.color_palette['primary']
        ))
        
        # Add line chart for average churn probability
        fig.add_trace(go.Scatter(
            x=segment_stats['segment'],
            y=segment_stats['Avg_Churn_Prob'] * 100,
            name='Avg Churn Probability (%)',
            yaxis='y2',
            line=dict(color=self.color_palette['danger'], width=3),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title_text="Customer Segmentation Analysis",
            title_x=0.5,
            xaxis_title="Customer Segments",
            yaxis=dict(title="Customer Count", side="left"),
            yaxis2=dict(title="Average Churn Probability (%)", side="right", overlaying="y"),
            height=500,
            font=dict(size=14),
            showlegend=True
        )
        
        return fig
    
    def create_performance_comparison_chart(self, model_performance: Dict) -> go.Figure:
        """
        Create model performance comparison chart
        """
        models = list(model_performance.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [model_performance[model][metric] * 100 for model in models]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=models,
                y=values,
                text=[f"{val:.1f}%" for val in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            title_x=0.5,
            xaxis_title="Models",
            yaxis_title="Score (%)",
            height=500,
            font=dict(size=14),
            barmode='group'
        )
        
        return fig
    
    def create_summary_report(self, df: pd.DataFrame, 
                            model_summary: Dict,
                            churn_column: str = 'churn',
                            probability_column: str = 'churn_probability') -> str:
        """
        Create a comprehensive summary report
        """
        total_customers = len(df)
        
        # Handle different churn column data types
        if churn_column in df.columns:
            try:
                if df[churn_column].dtype in ['int64', 'float64']:
                    churn_count = df[churn_column].sum()
                else:
                    # For categorical columns, count non-zero/True/Yes values
                    churn_count = len(df[df[churn_column].isin([1, '1', 'Yes', 'yes', 'YES', 'True', 'true', 'TRUE', 'Churned', 'churned'])])
            except:
                churn_count = 0
        else:
            churn_count = 0
            
        churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
        
        # Risk level distribution
        if 'risk_level' in df.columns:
            risk_distribution = df['risk_level'].value_counts().to_dict()
        else:
            risk_distribution = {}
        
        # Model performance
        best_model = model_summary.get('best_model', {})
        best_model_name = best_model.get('name', 'Unknown')
        best_model_metrics = best_model.get('metrics', {})
        
        report = f"""
        ## 📊 Lunarv Churn Prediction Analysis Report
        
        ### 📈 Dataset Overview
        - **Total Customers**: {total_customers:,}
        - **Churn Rate**: {churn_rate:.1f}%
        - **Churned Customers**: {churn_count:,}
        - **Loyal Customers**: {total_customers - churn_count:,}
        
        ### 🎯 Model Performance
        - **Best Model**: {best_model_name.replace('_', ' ').title()}
        - **Accuracy**: {best_model_metrics.get('accuracy', 0):.1%}
        - **F1 Score**: {best_model_metrics.get('f1_score', 0):.1%}
        - **Precision**: {best_model_metrics.get('precision', 0):.1%}
        - **Recall**: {best_model_metrics.get('recall', 0):.1%}
        
        ### 📚 Understanding the Metrics (In Simple Terms)
        
        **🔍 Accuracy ({best_model_metrics.get('accuracy', 0):.1%})**
        - **What it means**: Out of every 100 predictions, {best_model_metrics.get('accuracy', 0)*100:.0f} were correct
        - **Simple explanation**: This is like getting {best_model_metrics.get('accuracy', 0)*100:.0f} out of 100 questions right on a test
        - **Good range**: 80%+ is excellent, 70-80% is good, below 70% needs improvement
        
        **🎯 F1 Score ({best_model_metrics.get('f1_score', 0):.1%})**
        - **What it means**: The balance between finding churned customers and avoiding false alarms
        - **Simple explanation**: Think of it as the "sweet spot" between being too strict and too lenient
        - **Good range**: 75%+ is excellent, 60-75% is good, below 60% needs work
        
        **✅ Precision ({best_model_metrics.get('precision', 0):.1%})**
        - **What it means**: When we predict someone will churn, {best_model_metrics.get('precision', 0)*100:.0f}% of the time we're right
        - **Simple explanation**: How many of our "churn alerts" are actually true
        - **Good range**: 70%+ is good, below 60% means too many false alarms
        
        **🔄 Recall ({best_model_metrics.get('recall', 0):.1%})**
        - **What it means**: Out of all customers who actually churned, we caught {best_model_metrics.get('recall', 0)*100:.0f}% of them
        - **Simple explanation**: How many churning customers we didn't miss
        - **Good range**: 70%+ is good, below 60% means we're missing too many churners
        
        ### ⚠️ Risk Assessment
        """
        
        if risk_distribution:
            for risk_level, count in risk_distribution.items():
                percentage = (count / total_customers * 100) if total_customers > 0 else 0
                report += f"- **{risk_level}**: {count:,} customers ({percentage:.1f}%)\n"
        else:
            report += "- Risk assessment not available\n"
        
        report += f"""
        ### 💡 Key Insights
        - The model achieved **{best_model_metrics.get('accuracy', 0):.1%} accuracy** in predicting customer churn
        - **{churn_rate:.1f}%** of customers are at risk of churning
        - Early intervention strategies should focus on high-risk customers
        
        ### 🚀 Actionable Recommendations
        
        **🔥 High Priority (Immediate Action Required)**
        1. **Contact customers with >70% churn probability** within 24-48 hours
        2. **Personalized retention offers** for high-risk customers
        3. **Account manager intervention** for VIP customers at risk
        
        **⚠️ Medium Priority (Preventive Action)**
        1. **Implement retention programs** for customers with 30-70% churn probability
        2. **Regular check-ins** and satisfaction surveys
        3. **Proactive service improvements** based on usage patterns
        
        **✅ Low Priority (Maintenance)**
        1. **Continue monitoring** model performance monthly
        2. **Retrain model** when new data becomes available
        3. **Track retention success rates** to measure intervention effectiveness
        
        ### 📊 Business Impact
        - **Potential Revenue Saved**: By retaining high-risk customers, you could save significant revenue
        - **Customer Lifetime Value**: Focus on customers with highest potential long-term value
        - **ROI on Retention**: Retention programs typically cost 5-10x less than acquiring new customers
        
        ### 🔮 Next Steps
        1. **Export the results** to share with your team
        2. **Set up automated alerts** for new high-risk customers
        3. **Create retention campaigns** based on risk levels
        4. **Schedule monthly reviews** to track progress and adjust strategies
        """
        
        return report
