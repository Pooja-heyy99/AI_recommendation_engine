from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Recommendation Engine"
    env: str = "local"
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/recsys"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 120
    top_k_default: int = 10
    content_weight: float = 0.6
    aws_region: str = "us-east-1"
    ab_test_name: str = "retention_experiment"
    enable_aws_metrics: bool = False


settings = Settings()
