from flask import Flask
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoder from the pickle file
#def load_model():
  #global model
with open('motivation_model.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

# Function to predict motivational post based on user input
def predict_motivation_post(mood, age, profession, tone, industry):
    input_data = pd.DataFrame([[mood, age, profession, tone, industry]],
                              columns=['Mood', 'Age', 'Profession', 'Tone', 'Industry'])
    
    # Predict the motivational post
    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    return prediction[0]

# Get user input


# Predict and display the motivational post

#print(f"Predicted Motivational Post: {predicted_post}")

@app.route('/predict')
def predict():
    user_mood = "Happy"
    user_age = "<18"
    user_profession = "Doctor" 
    user_tone = "Positive"
    user_industry = "I1"
    predicted_post = predict_motivation_post(user_mood, user_age, user_profession, user_tone, user_industry)
    return predicted_post

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)