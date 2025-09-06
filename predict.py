# simple_predictor_dataframe.py
import joblib
import pandas as pd
import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text_with_spacy(text):
    """Text preprocessing function"""
    if isinstance(text, list):
        text = text[0]  # Extract text from list

    doc = nlp(text.lower())
    tokens = [
        token.lemma_.lower() for token in doc 
        if not token.is_stop 
        and not token.is_punct 
        and token.is_alpha 
        and len(token) > 2
    ]
    return ' '.join(tokens)

def predict_single_customer(input_df):
    """
    Predict for a single customer in DataFrame format
    """
    try:
        # Load models
        preprocessor = joblib.load('artifacts/structured_preprocessor.pkl')
        text_processor = joblib.load('artifacts/text_preprocessor.pkl')
        label_encoder = joblib.load('artifacts/label_encoder.pkl')
        model = joblib.load('saved_models/best_model.pkl')
        
        # Preprocess text
        processed_note = preprocess_text_with_spacy(input_df['Customer_Note'].iloc[0])
        
        # Transform features
        structured_features = preprocessor.transform(input_df)
        text_features = text_processor.transform([processed_note]).toarray()
        
        # Combine and predict
        features = np.hstack((structured_features, text_features))
        prediction = model.predict(features)
        proba = model.predict_proba(features)
        
        return {
            'tier': label_encoder.inverse_transform(prediction)[0],
            'confidence': float(np.max(proba)),
            'processed_note': processed_note,
            'success': True
        }
        
    except Exception as e:
        return {'error': str(e), 'success': False}

# # Example usage
# input_df = pd.DataFrame([{
#     "Gender": ['Male'],
#     "Driving_License": [1],
#     "Previously_Insured": [0],
#     "Vehicle_Age": ["1-2 Years"],
#     "Vehicle_Damage": ['Yes'],
#     "Region_Code": [28],
#     "Age": [45],
#     "Annual_Premium": [35000],
#     "Vintage": [200],
#     "Customer_Note": ["Customer called about insurance package."]
# }])

# result = predict_single_customer(input_df)
# if result['success']:
#     print(f"Tier: {result['tier']}, Confidence: {result['confidence']:.2%}")
# else:
#     print(f"Error: {result['error']}")

