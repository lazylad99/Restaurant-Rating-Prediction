# predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import joblib

class RestaurantRatingPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'rf') -> Dict[str, Any]:
        """
        Train model with hyperparameter tuning
        """
        print(f"Training {model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Select and tune model
        if model_type == 'rf':
            base_model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif model_type == 'dt':
            base_model = DecisionTreeRegressor(random_state=42)
            param_grid = {
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        else:  # Linear Regression
            base_model = LinearRegression()
            param_grid = {}
        
        if param_grid:
            print("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        else:
            self.model = base_model
            self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return {
            'metrics': metrics,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def visualize_results(self, results: Dict[str, Any]) -> None:
            """
            Create comprehensive visualizations with fixed plotting methods
            """
            print("Generating visualizations...")
            fig = plt.figure(figsize=(20, 10))
            
            # Actual vs Predicted
            ax1 = fig.add_subplot(231)
            ax1.scatter(results['y_test'], results['y_pred'], alpha=0.5)
            ax1.plot([results['y_test'].min(), results['y_test'].max()],
                    [results['y_test'].min(), results['y_test'].max()],
                    'r--', lw=2)
            ax1.set_xlabel('Actual Ratings')
            ax1.set_ylabel('Predicted Ratings')
            ax1.set_title('Actual vs Predicted Ratings')
            
            # Feature Importance - Fixed plotting method
            if hasattr(self.model, 'feature_importances_'):
                ax2 = fig.add_subplot(232)
                feature_importance = pd.DataFrame({
                    'feature': self.model.feature_names_in_,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                # Using barh from matplotlib instead of seaborn
                ax2.barh(feature_importance['feature'], feature_importance['importance'])
                ax2.set_title('Feature Importance')
                ax2.set_xlabel('Importance')
                ax2.set_ylabel('Features')
            
            # Rating Distribution
            ax3 = fig.add_subplot(233)
            sns.histplot(data=results['y_test'], bins=20, ax=ax3)
            ax3.set_title('Actual Rating Distribution')
            ax3.set_xlabel('Rating')
            ax3.set_ylabel('Count')
            
            # Prediction Error Distribution
            ax4 = fig.add_subplot(234)
            errors = results['y_test'] - results['y_pred']
            sns.histplot(data=errors, bins=20, ax=ax4)
            ax4.set_title('Prediction Error Distribution')
            ax4.set_xlabel('Prediction Error')
            ax4.set_ylabel('Count')
            
            # Add error percentages
            ax5 = fig.add_subplot(235)
            error_ranges = [0.25, 0.5, 0.75, 1.0]
            error_percentages = [
                (abs(errors) <= r).mean() * 100 for r in error_ranges
            ]
            ax5.bar(range(len(error_ranges)), error_percentages)
            ax5.set_xticks(range(len(error_ranges)))
            ax5.set_xticklabels([f'±{r:.2f}' for r in error_ranges])
            ax5.set_title('Prediction Error Ranges')
            ax5.set_xlabel('Error Range')
            ax5.set_ylabel('Percentage of Predictions')
            
            plt.tight_layout()
            plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model
        """
        self.model = joblib.load(filepath)

def main():
    try:
        # Load processed data
        print("Loading processed data...")
        df_processed = pd.read_csv('processed_data.csv')
        
        # Load feature columns from preprocessor
        preprocessor_state = joblib.load('preprocessor_state.joblib')
        feature_columns = preprocessor_state['feature_columns']
        
        # Prepare features
        X = df_processed[feature_columns]
        y = df_processed['rate']
        
        # Initialize and train predictor
        predictor = RestaurantRatingPredictor()
        results = predictor.train_model(X, y, model_type='rf')
        
         # Print metrics
        print("\nModel Performance Metrics:")
        print(f"MSE: {results['metrics']['mse']:.3f}")
        print(f"RMSE: {results['metrics']['rmse']:.3f}")
        print(f"MAE: {results['metrics']['mae']:.3f}")
        print(f"R²: {results['metrics']['r2']:.3f}")
        
        # Print additional performance insights
        print("\nPrediction Error Analysis:")
        errors = abs(results['y_test'] - results['y_pred'])
        print(f"Predictions within ±0.25 stars: {(errors <= 0.25).mean():.1%}")
        print(f"Predictions within ±0.5 stars: {(errors <= 0.5).mean():.1%}")
        print(f"Predictions within ±0.75 stars: {(errors <= 0.75).mean():.1%}")
        print(f"Predictions within ±1.0 stars: {(errors <= 1.0).mean():.1%}")
        
        # Visualize results
        predictor.visualize_results(results)
        
        # Save model
        print("\nSaving trained model...")
        predictor.save_model('trained_model.joblib')
        
        print("Model training and saving completed successfully!")
        
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()