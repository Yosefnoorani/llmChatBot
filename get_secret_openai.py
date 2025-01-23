import os
from dotenv import load_dotenv

# CORS(app)

# Load environment variables from the .env file
load_dotenv()

def get_secret():
    # def is_running_in_cloud():
    #     return os.getenv('RUNNING_IN_CLOUD') == 'true'
    #
    # # secret_key = None
    # if is_running_in_cloud():
    #     # Cloud-specific code
    #     print("Running in the cloud")
    #     secret_key = get_secret_cloud("GOOGLE_API_KEY")
    # else:
    #     # Local-specific code
    #     print("Running locally")
    #     load_dotenv()
    #     # secret_key = os.environ.get('GOOGLE_API_KEY')

    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Check if the API key is found
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        raise ValueError("API key not found in .env file")
    return api_key