from flask import Flask, render_template, request
import pickle

# Modeli yükle
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


app = Flask(__name__)

# Modeli yükleyin
model_path = r'C:\Users\brain\Desktop\seaborn_ML_models\penguin_type_prediction\penguin_species_model.pkl'
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data and preprocess it
    sex = request.form['sex']
    island = int(request.form['island'])
    bill_length = float(request.form['bill_length'])
    bill_depth = float(request.form['bill_depth'])
    flipper_length = float(request.form['flipper_length'])
    body_mass = float(request.form['body_mass'])    

    # Make predictions using the loaded model
    sample_data = [[sex, island, bill_length, bill_depth, flipper_length, body_mass]]
    prediction = model.predict(sample_data)

    # Determine the penguin species based on the prediction
    species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    predicted_species = species_mapping.get(prediction[0], 'Unknown')

    return render_template('result.html', predicted_species=predicted_species)

if __name__ == '__main__':
    app.run(debug=True)
