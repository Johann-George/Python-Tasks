from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):

    DB_USERNAME: str
    PASSWORD: str
    HOSTNAME: str
    PORT: int
    DATABASE: str

    model_config = SettingsConfigDict(
        env_file = str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding = "utf-8"
    )

    def database_url(self):
        return f"postgresql://{self.DB_USERNAME}:{self.PASSWORD}@{self.HOSTNAME}:{self.PORT}/{self.DATABASE}"