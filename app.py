from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

# Load the model during server startup
model = None
model_path = 'motivation_model.pkl'

def load_model():
    global model
    global label_encoder
    with open('motivation_model.pkl', 'rb') as f:
      model, label_encoder = pickle.load(f)

def predict_motivation_post(mood, age, profession, tone, industry):
    input_data = pd.DataFrame([[mood, age, profession, tone, industry]],
                              columns=['Mood', 'Age', 'Profession', 'Tone', 'Industry'])
    
    # Predict the motivational post
    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    return prediction[0]


@app.route('/predict')
def predict():
    user_mood = request.args.get('mood')
    user_age = request.args.get('age')
    user_profession = request.args.get('prfession') 
    user_tone = request.args.get('tone')
    user_industry = request.args.get('industry')
    predicted_post = predict_motivation_post(user_mood, user_age, user_profession, user_tone, user_industry)
    return jsonify({"data": predicted_post}) 




@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
