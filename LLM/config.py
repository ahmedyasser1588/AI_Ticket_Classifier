from pydantic import Field
from pydantic_settings import BaseSettings,SettingsConfigDict

class LLMConfig(BaseSettings):
    api_key: str = Field(..., env="api_key")
    base_url: str = Field(..., env="base_url")
    model: str = Field(..., env="model")
    timeout:float=Field(30.0,ge=1.0)
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="LLM_"
    )
