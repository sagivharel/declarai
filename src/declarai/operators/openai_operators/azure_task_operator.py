import logging
from functools import partial
from typing import Callable, List, Optional, Type

from declarai.operators.base.types import Message
from .base_task_operator import BaseOpenAITaskOperator
from .openai_llm import AzureOpenAILLM

logger = logging.getLogger("OpenAITaskOperator")


class AzureOpenAITaskOperator(BaseOpenAITaskOperator):
    llm: AzureOpenAILLM

    @classmethod
    def new_operator(
        cls,
        openai_token: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Type["AzureOpenAITaskOperator"]:
        openai_llm = AzureOpenAILLM(openai_token, model)
        partial_class = partial(cls, openai_llm)
        return partial_class
