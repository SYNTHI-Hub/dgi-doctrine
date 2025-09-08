"""
import requests

url = "http://127.0.0.1:8000/o/token/"
data = {
    "grant_type": "password",
    "username": "vinsd95@gmail.com",
    "password": "admin@1234",
    "client_id": "nCgjoDdiRDHCyq3JYCXCBork4vDmkXcAaj8C0Kqr",
    "client_secret": "4F11Z0HTO69j2URg8nP2xqvFH4fr8N9ZdsPbMpXLEjkLEAQKa28W45shZDLwN1PAJpo4TRd511R5Ukk8LGkqer5dekGsjzkHv6g98JApUvBuX3pRYIh7ANXYZwBIzuau",
    "scope": "read write"
}
headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}

response = requests.post(url, data=data, headers=headers)
print(response.json())
"""


import os
import requests
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


if not client_id or not client_secret:
    raise ValueError("CLIENT_ID or CLIENT_SECRET missing in .env file")


url = "http://127.0.0.1:8000/o/token/"
data = {
    "grant_type": "password",
    "username": "vinsd95@gmail.com",
    "password": "admin@1234",
    "client_id": client_id,
    "client_secret": client_secret,
    "scope": "read write"
}
headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}


response = requests.post(url, data=data, headers=headers)
print(response.json())