from typing import Any 

from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse

from src.config import settings
from src.logger import get_console_logger
from src.inference_pipeline.app.endpoints import api_router


root_router = APIRouter()

app = FastAPI(
  title=settings.comet_project_name,
  openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@root_router.get("/")
def index(request:Request) -> Any:
  
  body = (
    "<html>"
    "<body style='padding: 10px;'>"
    "<h1>Welcome to the Model API</h1>"
    "<div>"
    "Check the docs: <a href='/docs'>here</a>"
    "</div>"
    "</body>"
    "</html>"
  )
  
  return HTMLResponse(content=body)


app.include_router(
  router=api_router, 
  prefix=settings.API_V1_STR
)

app.include_router(router=root_router)


if __name__ == "__main__":
  
  logger = get_console_logger()
  
  import uvicorn
  
  # Start Uvicorn web server 
  uvicorn.run(  
    app=app,
    host="localhost",
    port=8001,
    log_level="debug"
  )
  