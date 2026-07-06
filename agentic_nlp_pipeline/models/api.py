from dotenv import load_dotenv
import os


class APIModel:
    """Wrapper for language models called over an API."""

    # Load environment variables from the `.env` file.
    load_dotenv()

    # TODO: Complete this method
    def __init__(self):
        self.api_key = os.getenv("API_KEY")

    # TODO: Complete this method
    def generate(self) -> str: ...
