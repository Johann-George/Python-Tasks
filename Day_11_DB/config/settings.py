from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    #database config 
    DB_USERNAME: str
    PASSWORD: str
    HOSTNAME: str
    PORT: int
    DATABASE: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def database_url(self):
        return f"mysql+pymysql://{self.DB_USERNAME}:{self.PASSWORD}@{self.HOSTNAME}:{self.PORT}/{self.DATABASE}"

# settings = Settings    