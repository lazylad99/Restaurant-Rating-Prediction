import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict
import joblib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings


class RestaurantDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data processing pipeline
        """
        df_processed = df.copy()

        print("Starting data preprocessing...")

        # Handle missing values
        print("Handling missing values...")
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns

        for col in numeric_cols:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)

        for col in categorical_cols:
            mode_value = df_processed[col].mode()[0]
            df_processed[col] = df_processed[col].fillna(mode_value)

        # Remove duplicates
        print("Removing duplicates...")
        df_processed = df_processed.drop_duplicates(keep='first')

        # Feature engineering
        print("Performing feature engineering...")
        if 'cuisines_type' in df_processed.columns:
            df_processed['cuisine_count'] = df_processed['cuisines_type'].str.count(',') + 1
        else:
            df_processed['cuisine_count'] = 1  # Default value if missing

        # Create price categories safely
        try:
            df_processed['price_category'] = pd.qcut(
                df_processed['avg_cost'], q=4, labels=['Budget', 'Medium', 'High', 'Premium']
            )
        except ValueError:
            df_processed['price_category'] = 'Medium'  # Default if qcut fails

        # Encode categorical variables
        print("Encoding categorical variables...")
        categorical_features = ['restaurant_type', 'area', 'cuisines_type', 'price_category']
        for col in categorical_features:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])

        # Convert binary features
        binary_features = ['online_order', 'table_booking']
        for col in binary_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

        # Scale numerical features
        print("Scaling numerical features...")
        numerical_features = ['num_of_ratings', 'avg_cost', 'cuisine_count']
        df_processed[numerical_features] = self.scaler.fit_transform(df_processed[numerical_features])

        print("Preprocessing completed!")
        return df_processed

    def prepare_features(self, df_processed: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable
        """
        self.feature_columns = [
            'num_of_ratings', 'avg_cost', 'online_order', 'table_booking',
            'restaurant_type', 'area', 'cuisine_count', 'price_category'
        ]

        X = df_processed[self.feature_columns]
        y = df_processed['rate'] if 'rate' in df_processed.columns else None

        return X, y

    def save_preprocessor(self, filepath: str) -> None:
        """
        Save preprocessor state
        """
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)

    def load_preprocessor(self, filepath: str) -> None:
        """
        Load preprocessor state
        """
        saved_state = joblib.load(filepath)
        self.label_encoders = saved_state['label_encoders']
        self.scaler = saved_state['scaler']
        self.feature_columns = saved_state['feature_columns']

    def transform_new_data(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data for prediction
        """
        df_new = df_new.copy()

        # Handle missing values
        numeric_cols = df_new.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_new.select_dtypes(include=['object']).columns

        for col in numeric_cols:
            df_new[col] = df_new[col].fillna(df_new[col].median())

        for col in categorical_cols:
            df_new[col] = df_new[col].fillna(df_new[col].mode()[0])

        # Feature engineering
        if 'cuisines_type' in df_new.columns:
            df_new['cuisine_count'] = df_new['cuisines_type'].str.count(',') + 1
        else:
            df_new['cuisine_count'] = 1

        if 'avg_cost' in df_new.columns:
            df_new['price_category'] = pd.qcut(df_new['avg_cost'], q=4, labels=['Budget', 'Medium', 'High', 'Premium'])
        else:
            df_new['price_category'] = 'Medium'

        # Encode categorical variables
        categorical_features = ['restaurant_type', 'area', 'cuisines_type', 'price_category']
        for col in categorical_features:
            if col in df_new.columns and col in self.label_encoders:
                df_new[col] = self.label_encoders[col].transform(df_new[col])

        # Convert binary features
        binary_features = ['online_order', 'table_booking']
        for col in binary_features:
            if col in df_new.columns:
                df_new[col] = df_new[col].map({'Yes': 1, 'No': 0})

        # Scale numerical features
        numerical_features = ['num_of_ratings', 'avg_cost', 'cuisine_count']
        df_new[numerical_features] = self.scaler.transform(df_new[numerical_features])

        return df_new[self.feature_columns]


def main():
    try:
        print("Loading data...")
        df = pd.read_csv('restaurants.csv')

        preprocessor = RestaurantDataPreprocessor()

        df_processed = preprocessor.process_data(df)
        X, y = preprocessor.prepare_features(df_processed)

        print("Saving processed data...")
        df_processed.to_csv('processed_data.csv', index=False)
        preprocessor.save_preprocessor('preprocessor_state.joblib')

        print("Data preprocessing and saving completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
