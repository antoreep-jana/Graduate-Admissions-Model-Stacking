# -*- coding: utf-8 -*-


from flask import Flask, request, jsonify, render_template

import numpy as np
import pickle
import joblib

app = Flask(__name__)

# not executed
#model = pickle.load(open('model.pkl', 'rb'))

def load_model(name):
    print(f"Loading {name}...")
    return pickle.load(open(name, 'rb'))
    

level_1_model_names = ['model1.pkl','model2.pkl','model3.pkl', 'model4.pkl','model5.pkl','model6.pkl','model7.pkl','model8.pkl','model9.pkl','model10.pkl','model11.pkl','model12.pkl','model13.pkl','model14.pkl']

level_1_models = []
for model in level_1_model_names:
    level_1_models.append(load_model(model))

stage_1_ensemble_5_model_names = ['ensemble_model1.pkl','ensemble_model2.pkl','ensemble_model3.pkl','ensemble_model4.pkl','ensemble_model5.pkl']

ensemble_level_1_models = []
for model in stage_1_ensemble_5_model_names:
    ensemble_level_1_models.append(load_model(model))

stage_2_model_names = ['stage_2_model1.pkl','stage_2_model2.pkl','stage_2_model3.pkl']
stage_2_models = []

for model in stage_2_model_names:
    stage_2_models.append(load_model(model))
    
meta_model = load_model('meta_model.pkl')

scaler = joblib.load('scaler.save')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    
    # Perform feature scaling for test data
    
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    final_features = scaler.transform(final_features)
    
    features_level_1 = np.zeros(shape = (1, len(level_1_models)))
    for i, clf in enumerate(level_1_models):
        features_level_1[:, i] = clf.predict(final_features)
    
    
    features_level_2 = np.zeros(shape = (1, 5))
    for i, clf in enumerate(ensemble_level_1_models):
        features_level_2[:,i] = clf.predict(features_level_1)
        
    features_level_3 = np.zeros(shape = (1,3))
    for i, clf in enumerate(stage_2_models):
        features_level_3[:,i] = clf.predict(features_level_2)
    
    #prediction = model.predict(final_features)
    
    prediction = meta_model.predict(features_level_3)
    out = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text = f'Chances of admit should be {out*100} %')

if __name__ == "__main__":
    app.run(debug = True)