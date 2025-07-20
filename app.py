from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('movie_rating_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        runtime = float(request.form['runtime'])
        metascore = float(request.form['metascore'])
        votes = int(request.form['votes'])

        features = np.array([[runtime, metascore, votes]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f'Predicted IMDB Rating: {prediction:.2f}')
    except:
        return render_template('index.html', prediction_text='⚠️ Invalid input! Please try again.')

if __name__ == '__main__':
    app.run(debug=True)