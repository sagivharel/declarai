import logging
from functools import partial
from typing import List, Optional, Type
from typing_extensions import Self

from declarai.operators.base.types import Message
from .base_chat_operator import BaseOpenAIChatOperator

from .openai_llm import OpenAILLM

logger = logging.getLogger("OpenAIChatOperator")


class OpenAIChatOperator(BaseOpenAIChatOperator):
    llm: OpenAILLM
    compiled_template: List[Message]

    @classmethod
    def new_operator(
        cls,
        openai_token: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Type["OpenAIChatOperator"]:
        openai_llm = OpenAILLM(openai_token, model)
        partial_class: Self = partial(cls, openai_llm)
        return partial_class
