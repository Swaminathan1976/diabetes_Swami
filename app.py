import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('DiabetesV2.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    pregnancies = float(request.form['Pregnancies'])
    glucose = float(request.form['Glucose'])
    bloodpressure = float(request.form['Blood Pressure'])
    skinthickness = float(request.form['Skin Thickness'])
    insulin = float(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    diabetespedigreefunction = float(request.form['Diabetes Pedigree Function'])
    age=float(request.form['Age'])
	
    finalFeatures = np.array([[pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,age]])
    prediction = model.predict(finalFeatures)
    if prediction==1:
        return render_template('index.html', prediction_text = "You have diabetes")
    else:
        return render_template('index.html', prediction_text= "You don't have diabetes")
    

    return render_template('index.html', prediction_text='Diabetes Outcome is  $ {}'.format(round(prediction[0][0])))


if __name__ == "__main__":
    app.run(debug=True)