from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  
  API_V1_STR: str = "/api/v1"
  
  model_config = SettingsConfigDict(
    env_file=".env", 
    env_file_encoding="utf-8",
    extra="allow"
  )
  
  # Polygon
  polygon_api_key: str
  
  # CometML
  comet_api_key: str
  comet_workspace: str
  comet_model_name: str
  comet_project_name: str
  
  # Cerebrium
  cerebrium_api_key: str
  
  
settings = Settings()
