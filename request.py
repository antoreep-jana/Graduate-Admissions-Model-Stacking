# -*- coding: utf-8 -*-

import requests

url = 'http://localhost:5000/predict_api'

r = requests.post(url, json = {'GRE Score' : 330, 'TOEFL Score' : 110, 'University Ranking' : 8.5, 'SOP' : 4.5, 'LOR' : 4, 'CGPA' : 8.28, 'Research' : 1})

print(r.json())


## FIX THE DEPLOYMENT REQUIREMENTS FILE