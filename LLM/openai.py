from .base import LLM
from openai import OpenAI
from .Model import GenerationRequest
from .config import LLMConfig
from .exceptions import (
    ConnectionError,
    ProvideError,
    InvalidRequestError,
    LLMError,
    AuthError,
    RateLimitError)
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeOutError,
    AuthenticationError as OpenAIAuthError ,
    BadRequestError,
    RateLimitError as OpenAIRateLimitError
     ) 
class OpenAiCompatible(LLM):
    def __init__(
        self,
        settings: LLMConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        if isinstance(settings, str):
            api_key = settings
            settings = None

        config = settings or LLMConfig()
        self.model = model or config.model
        self.client = OpenAI(
            api_key=api_key or config.api_key,
            base_url=base_url or config.base_url,
        )

    def generate(self, request: GenerationRequest):
        messages = [{
            "role": message.role.value,
            "content": message.content
        } for message in request.messages]  
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_new_tokens,
                temperature=request.temp if request.temp is not None else request.temperature,
            top_p=request.top_p
        )
        except APIConnectionError:
            raise ConnectionError("Failed to connect to the LLM API")
        except APITimeOutError:
            raise ConnectionError("LLM server is sleeping ")
        except OpenAIAuthError:
            raise AuthError("Authentication token is not correct "
                            "or Model is not found")
        except BadRequestError:
            raise InvalidRequestError("Request is broken... ")
        except OpenAIRateLimitError:
            raise RateLimitError("you have comsumed your quota")
        except Exception as e:
            raise LLMError(str(e))

        return response.choices[0].message.content
