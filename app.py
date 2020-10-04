{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'tensorflow'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-1437567f5b63>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mflask\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mFlask\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrequest\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mjsonify\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrender_template\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[1;32mimport\u001b[0m \u001b[0mtensorflow\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mapp\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mFlask\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0m__name__\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'tensorflow'"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from flask import Flask, request, jsonify, render_template\n",
    "import tensorflow as tf\n",
    "\n",
    "app = Flask(__name__)\n",
    "model = tf.keras.models.load_model(\n",
    "    'Diabetesv2.h5', custom_objects=None, compile=True, options=None\n",
    ")\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "    '''\n",
    "    For rendering results on HTML GUI\n",
    "    '''\n",
    "    pregnancies = float(request.form['Pregnancies'])\n",
    "    glucose = float(request.form['Glucose'])\n",
    "    bloodpressure = float(request.form['Blood Pressure'])\n",
    "    skinthickness = float(request.form['Skin Thickness'])\n",
    "    insulin = float(request.form['Insulin'])\n",
    "    bmi = float(request.form['BMI'])\n",
    "    diabetespedigreefunction = float(request.form['Diabetes Pedigree Function'])\n",
    "    age=float(request.form['Age'])\n",
    "\t\n",
    "    finalFeatures = np.array([[pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,age]])\n",
    "    prediction = model.predict(finalFeatures)\n",
    "    if prediction==1:\n",
    "        return render_template('index.html', prediction_text = \"You have diabetes\")\n",
    "    else:\n",
    "        return render_template('index.html', prediction_text= \"You don't have diabetes\")\n",
    "    \n",
    "\n",
    "    return render_template('index.html', prediction_text='Diabetes Outcome is  $ {}'.format(round(prediction[0][0])))\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
