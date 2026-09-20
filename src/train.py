import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    print("Loading data...")
    # Dynamically getting the path to data.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'data.csv')
    
    df = pd.read_csv(data_path)

    print("Cleaning data...")
    df = df.drop_duplicates()
    
    # Clean string columns (remove trailing spaces)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # Drop leaky and unnecessary columns
    cols_to_drop = ['Unnamed: 13', 'Feedback', 'Pin code', 'latitude', 'longitude']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Separate Features and Target
    X = df.drop(columns=['Output'])
    y = df['Output'].apply(lambda x: 1 if x == 'Yes' else 0)

    print("Building model pipeline...")
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ], 
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    print("Training the model on full dataset...")
    # Production me hum poore data par train karte hain
    model_pipeline.fit(X, y)

    # Save the model
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True) # Create folder if it doesn't exist
    model_path = os.path.join(model_dir, 'model.pkl')
    
    joblib.dump(model_pipeline, model_path)
    print(f"Success! Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()