import base64
import hashlib
import os
import random
import string

from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("CLIENT_ID")
secret =os.getenv("CLIENT_SECRET")
credential = "{0}:{1}".format(client_id, secret)
cred = base64.b64encode(credential.encode("utf-8"))

print(cred)




code_verifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(random.randint(43, 128)))

code_challenge = hashlib.sha256(code_verifier.encode('utf-8')).digest()
code_challenge = base64.urlsafe_b64encode(code_challenge).decode('utf-8').replace('=', '')

print("code challenge:")
print(code_challenge)
print(code_verifier)