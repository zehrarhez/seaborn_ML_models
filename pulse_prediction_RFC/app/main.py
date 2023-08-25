from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Modeli yükle
model = joblib.load(r'C:\Users\brain\Desktop\seabornLibs\exercise\model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Formdan gelen verileri al
    kind = int(request.form.get('kind'))
    time = int(request.form.get('time'))
    diet = int(request.form.get('diet'))

    # Verileri modele uygun hale getir
    sample = [[kind, time, diet]]

    # Tahmin yap
    prediction = model.predict(sample)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
