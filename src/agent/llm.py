"""A LangChain chat model backed by the project's metered API client.

``ApiClient`` already caches answers, prices them at the provider's peak rate,
refuses a request once the budget is spent and keeps local models on this
machine.  This class adds nothing to that contract: it translates LangChain
messages to the client's message format and wraps the answer in an
``AIMessage``, so the graph talks to models through one HTTP client only.
"""

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from src.model.api_client import ApiClient

ROLES = {'human': 'user', 'system': 'system', 'ai': 'assistant'}


class ApiChatModel(BaseChatModel):
    """The chat model of the pipeline: one metered, cached ``ApiClient`` call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: ApiClient
    max_tokens: int = 8000

    @property
    def _llm_type(self) -> str:
        return 'api-' + self.client.model

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs) -> ChatResult:
        payload = [{'role': ROLES[message.type], 'content': message.content}
                   for message in messages]
        answer = self.client.chat(payload, max_tokens=self.max_tokens)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer['content']))])
