import openai
from .base_openai_llm import BaseOpenAILLM
from .settings import OPENAI_API_KEY, OPENAI_MODEL


class AzureOpenAIError(Exception):
    pass


class AzureOpenAILLM(BaseOpenAILLM):
    def __init__(
        self,
        openai_token: str = None,
        model: str = None

    ):
        self.openai = openai
        self.openai.api_key = openai_token or OPENAI_API_KEY
        if not self.openai.api_key:
            raise AzureOpenAIError(
                "Missing an OpenAI API key"
                "In order to work with OpenAI, you will need to provide an API key"
                "either by setting the DECLARAI_OPENAI_API_KEY or by providing"
                "the API key via the init interface."
            )
        self.model = model or OPENAI_MODEL
