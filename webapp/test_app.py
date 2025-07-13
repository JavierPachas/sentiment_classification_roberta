import requests
import json

# The URL of your Flask app's prediction endpoint
url = "http://127.0.0.1:8000/predict"

# The data you want to send in the POST request
data = {
    "text": "I was very disappointed with the service."
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response from the server
print(response.status_code)
print(json.dumps(response.json(), indent=4))