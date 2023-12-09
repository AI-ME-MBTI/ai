import deepl

from starlette.config import Config
from dotenv import load_dotenv

load_dotenv()

config = Config("./.env")
AUTH_KEY = config.get('AUTH_KEY')

def get_translate(text):
    auth_key = AUTH_KEY
    translator = deepl.Translator(auth_key)

    result = translator.translate_text(text, target_lang="EN-US")
    print(result)
    return str(result)
