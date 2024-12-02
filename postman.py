import requests

url = 'http://127.0.0.1:5000/predict'
file_path = 'https://storage.googleapis.com/model_agri-scan/healthy.jpg'

with open(file_path, 'rb') as img_file:
    response = requests.post(url, files={'file': img_file})

print(response.json())
