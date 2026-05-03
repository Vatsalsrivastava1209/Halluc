from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")
    
    models_to_test: list[str] = Field(
        default=["gpt-4o-mini", "claude-3-5-sonnet-20240620"],
        description="Add more models here anytime"
    )
    categories: list[str] = Field(
        default=["technology", "science", "politics", "business", "sports"],
        description="News categories to track daily"
    )
    judge_model: str = "gpt-4o"
    max_tokens: int = 150

settings = Settings()