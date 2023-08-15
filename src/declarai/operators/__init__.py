from typing import Type, Union

from typing_extensions import Literal

from declarai.operators.base.llm_settings import LLMSettings
from declarai.operators.base.types.operator import BaseOperator
from .base.types.llm_params import LLMParamsType

from .openai_operators.chat_operator import OpenAIChatOperator
from .openai_operators.task_operator import OpenAITaskOperator

# Based on documentation from https://platform.openai.com/docs/models/overview
ProviderOpenai = Literal["openai"]
ModelsOpenai = Literal[
    "gpt-4",
    "gpt-3.5-turbo",
    "text-davinci-003",
    "text-davinci-002",
    "code-davinci-002",
]

AllModels = Union[ModelsOpenai]


def resolve_operator(
    llm_config: LLMSettings, operator_type: Literal["chat", "task"] = "task", **kwargs
) -> Type[BaseOperator]:
    """
    Resolves the operator to be used for the given llm_config
    :param llm_config: llm settings like provider, model, etc
    :param operator_type: relevant operator type
    :param kwargs: api keys, etc
    :return: a class that inherits from BaseOperator
    """
    if llm_config.provider == "openai":
        open_ai_token = kwargs.get("openai_token")
        model = llm_config.model
        if operator_type == "task":
            operator = OpenAITaskOperator
        elif operator_type == "chat":
            operator = OpenAIChatOperator
        else:
            raise NotImplementedError(
                f"Operator type : {operator_type} not implemented"
            )
        if open_ai_token:
            return operator.new_operator(openai_token=open_ai_token, model=model)
        return operator.new_operator(model=model)

    elif llm_config.provider == "azure_openai":
        azure_open_ai_version = kwargs.get("openai_api_version")
        azure_open_ai_resource = kwargs.get("openai_resource")
        azure_open_ai_key = kwargs.get("open_ai_key")
        if operator_type == "task":
            operator = OpenAITaskOperator
        elif operator_type == "chat":
            operator = OpenAIChatOperator

        if azure_open_ai_key:
            return operator.new_operator(
                openai_api_version=azure_open_ai_version,
                openai_resource=azure_open_ai_resource,
                openai_token=azure_open_ai_key,
            )



    raise NotImplementedError()
