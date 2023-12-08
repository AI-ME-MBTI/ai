import requests

from starlette.config import Config
from dotenv import load_dotenv

load_dotenv()

config = Config("./.env")
CLIENT_ID_SI = config.get('CLIENT_ID_SI')
CLIENT_SECRET_SI = config.get('CLIENT_SECRET_SI')

CLIENT_ID_HS = config.get('CLIENT_ID_HS')
CLIENT_SECRET_HS = config.get('CLIENT_SECRET_HS')

import requests

def get_translate(text, client_id = CLIENT_ID_SI, client_secret=CLIENT_SECRET_SI):
    
    data = {'text' : text,
            'source' : 'ko',
            'target': 'en'}

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id":client_id,"X-Naver-Client-Secret":client_secret}

    response = requests.post(url, headers=header, data=data)
    rescode = response.status_code

    if(rescode==200):
        send_data = response.json()
        trans_data = (send_data['message']['result']['translatedText'])
        return trans_data
    else:
        print("Error Code:" , rescode)
        get_translate(text, CLIENT_ID_HS,CLIENT_SECRET_SI)
