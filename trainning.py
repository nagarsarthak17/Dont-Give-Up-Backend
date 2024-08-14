import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the CSV data
data = pd.read_csv("don'tgiveup - Sheet1.csv")

# Define features and target variable
X = data[['Mood', 'Age', 'Profession', 'Tone', 'Industry']]
y = data['Post']  # Assuming the column for motivational posts is named 'MotivationPost'

# Encode the target variable if it's not numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define categorical features
categorical_features = ['Mood', 'Age', 'Profession', 'Tone', 'Industry']

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline with preprocessing and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model and label encoder as a pickle file
with open('motivation_model.pkl', 'wb') as f:
    pickle.dump((model, label_encoder), f)