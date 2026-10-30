class LLMError(Exception):
    pass

class AuthError(LLMError):
    pass

class InvalidRequestError(LLMError):
    pass

class ConnectionError(LLMError):
    pass

class ProvideError(LLMError):
    pass
class RateLimitError(LLMError):
    pass

