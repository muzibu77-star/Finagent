"""A LangChain chat model backed by the project's metered API client.

``ApiClient`` already caches answers, prices them at the provider's peak rate,
refuses a request once the budget is spent and keeps local models on this
machine.  This class adds nothing to that contract: it translates LangChain
messages to the client's message format and wraps the answer in an
``AIMessage``, so the graph and the single-agent control group talk to models
through one HTTP client only.  Tools are bound in the OpenAI format the client
sends, and tool calls travel both ways: requests carry an ``AIMessage``'s calls
and ``ToolMessage`` results, answers come back as an ``AIMessage`` with calls.

``on_delta`` is the one extra the graph may hand in, as a call keyword, when it
wants to show the text of an answer while it is written; it is forwarded to the
client, which decides whether it can stream.  A call without it sends exactly
the request it sent before.
"""

import json

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from src.model.api_client import ApiClient

ROLES = {'human': 'user', 'system': 'system', 'ai': 'assistant'}


def message_payload(message: BaseMessage) -> dict:
    """One LangChain message in the client's format, tool calls included."""
    if isinstance(message, ToolMessage):
        return {'role': 'tool', 'tool_call_id': message.tool_call_id,
                'content': content_text(message.content)}
    payload = {'role': ROLES[message.type], 'content': content_text(message.content)}
    calls = getattr(message, 'tool_calls', None)
    if calls:
        payload['tool_calls'] = [{'id': call['id'], 'type': 'function',
                                  'function': {'name': call['name'],
                                               'arguments': json.dumps(call['args'],
                                                                       ensure_ascii=False)}}
                                 for call in calls]
    return payload


def content_text(content) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def answer_message(answer: dict) -> AIMessage:
    """The client's answer as an ``AIMessage``, its tool calls decoded for the graph."""
    return AIMessage(content=answer['content'], tool_calls=[tool_call(call) for call in
                                                            (answer.get('tool_calls') or [])])


def tool_call(call: dict) -> dict:
    try:
        args = json.loads(call['function'].get('arguments') or '{}')
    except json.JSONDecodeError:
        args = {}  # the model gets a tool error for the missing arguments
    return {'name': call['function']['name'], 'args': args, 'id': call.get('id'),
            'type': 'tool_call'}


class ApiChatModel(BaseChatModel):
    """The chat model of the pipeline: one metered, cached ``ApiClient`` call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: ApiClient
    max_tokens: int = 8000
    used: dict = Field(default_factory=dict)  # prompt/completion tokens of this model's calls

    @property
    def _llm_type(self) -> str:
        return 'api-' + self.client.model

    def bind_tools(self, tools, **kwargs):
        """The model with an OpenAI-format tool list in every request."""
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs) -> ChatResult:
        payload = [message_payload(message) for message in messages]
        answer = self.client.chat(payload, max_tokens=self.max_tokens, tools=kwargs.get('tools'),
                                  on_delta=kwargs.get('on_delta'))
        for name in ('prompt_tokens', 'completion_tokens'):
            self.used[name] = self.used.get(name, 0) + answer['usage'][name]
        return ChatResult(generations=[ChatGeneration(message=answer_message(answer))])
