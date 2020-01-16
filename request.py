import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'data': 'hola como estas'})

print(r.json())