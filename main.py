import base64
import hashlib
import os
import secrets
import string
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

if not client_id or not client_secret:
    raise ValueError("CLIENT_ID ou CLIENT_SECRET manquant dans le fichier .env")

characters = string.ascii_letters + string.digits + "-._~"
code_verifier = ''.join(secrets.choice(characters) for _ in range(64))  # Longueur fixe de 64

code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier.encode('utf-8')).digest()
).decode('utf-8').rstrip('=')

print("Client ID:", client_id)
print("Client Secret:", client_secret)
print("Code Verifier:", code_verifier)
print("Code Challenge:", code_challenge)

credential = f"{client_id}:{client_secret}"
cred = base64.b64encode(credential.encode('utf-8')).decode('utf-8')
print("Basic Auth Header:", f"Basic {cred}")

# Instructions pour la requête /o/authorize/
authorize_url = (
    f"http://127.0.0.1:8000/o/authorize/?response_type=code"
    f"&code_challenge={code_challenge}&code_challenge_method=S256"
    f"&client_id={client_id}&redirect_uri=http://127.0.0.1:8000/admin/"
)
print("Authorize URL:", authorize_url)