from pydantic_settings import BaseSettings, SettingsConfigDict
from src.paths import PARENT_DIR

class GeneralConfig(BaseSettings):

  model_config = SettingsConfigDict(env_file=PARENT_DIR/".env", env_file_encoding="utf-8", extra="allow")
  
  exchange_rate_api_key: str
  
  # CometML
  comet_api_key: str
  comet_workspace: str
  comet_project_name: str

config = GeneralConfig()
