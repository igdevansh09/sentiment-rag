import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def hugging_face_auth():
    HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")
    login(token=HUGGING_FACE_TOKEN)
    print("Login Succesfull !!")
        
hugging_face_auth()        