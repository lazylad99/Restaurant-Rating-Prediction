import joblib
import pandas as pd
import numpy as np

def load_preprocessor_and_model():
    """
    Load the preprocessor and model from joblib files.
    """
    try:
        preprocessor = joblib.load('preprocessor_state.joblib')
        model = joblib.load('trained_model.joblib')
        return preprocessor, model
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor or model: {e}")

def preprocess_input_data(restaurant_data: dict, preprocessor: dict) -> pd.DataFrame:
    """
    Preprocess input restaurant data to match model requirements.
    """
    try:
        df = pd.DataFrame([restaurant_data])
        
        label_encoders = preprocessor['label_encoders']
        scaler = preprocessor['scaler']
        feature_columns = preprocessor['feature_columns']
        
        categorical_columns = ['restaurant_type', 'area', 'cuisines_type', 'price_category']
        numerical_columns = ['num_of_ratings', 'avg_cost', 'cuisine_count']
        
        # Encode categorical columns
        for col in categorical_columns:
            if col in label_encoders:
                df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)
        
        # Convert boolean-like columns to int (if applicable)
        df['online_order'] = df['online_order'].astype(int)
        df['table_booking'] = df['table_booking'].astype(int)
        
        # Scale numerical columns
        df[numerical_columns] = scaler.transform(df[numerical_columns])
        
        # Ensure correct feature order
        df = df[feature_columns]
        
        return df
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

def predict_restaurant_rating(restaurant_data: dict) -> float:
    """
    Predict restaurant rating given its details.
    """
    try:
        preprocessor, model = load_preprocessor_and_model()
        processed_data = preprocess_input_data(restaurant_data, preprocessor)
        prediction = model.predict(processed_data)[0]
        return prediction
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

def main():
    """
    Main function to test restaurant rating prediction.
    """
    restaurant = {
        'restaurant_type': 'Casual Dining',
        'cuisines_type': 'North Indian, Chinese',
        'area': 'Banashankari',
        'avg_cost': 800,
        'table_booking': 1,
        'online_order': 1,
        'num_of_ratings': 156,
        'cuisine_count': 2,
        'price_category': 'Medium'
    }
    
    try:
        predicted_rating = predict_restaurant_rating(restaurant)
        print(f"\nPredicted Rating: {predicted_rating:.1f}/5.0")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
