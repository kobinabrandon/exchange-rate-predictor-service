import pickle 
from comet_ml import API

from src.config import settings
from src.logger import get_console_logger
from src.paths import MODELS_DIR
from sklearn.pipeline import Pipeline


logger = get_console_logger()

def load_model_from_registry(
    model_name: str,
    status: str = "Production",
    api_key: str = settings.comet_api_key,
    workspace: str = settings.comet_workspace
) -> Pipeline:
    
    """ 
    Find all the versions of the relevant model, and choose the versions that are of the 
    appropriate status. Then extract the first of these versions. This version of this 
    model is the one that will be downloaded from CometML's model registry.
    
    You may find the documentation here:
    https://www.comet.com/docs/v2/guides/model-management/using-model-registry/
    """
    
    # Find the version of the model
    api = API(api_key)
    
    # Find the model versions
    model_details = api.get_registry_model_details(workspace=workspace, registry_name=model_name)["versions"]
    
    # Search the dictionary to extract the versions of the models that are of the relevant status 
    model_versions = [
        detail["version"] for detail in model_details if detail["status"] == status
    ]
    
    if len(model_versions) == 0:
        
        logger.error(f"No {status} model found")
        raise ValueError(f"No {status} model found")
    
    else:
        logger.info(f"Found these {status} model versions: {model_versions}")
        model_version = model_versions[0]

    
    # Download the model from the registry and put it in a local file
    api.download_registry_model(
        workspace = workspace,
        registry_name=model_name,
        version=model_version,
        output_path=MODELS_DIR,
        expand=True
    )
    
    # Load said local file and return it
    with open(MODELS_DIR/f"Tuned {model_name} model.pkl", "rb") as f:
        
        model = pickle.load(f)
        
    return model
        
        
if __name__ == "__main__":
    
    load_model_from_registry(
        model_name="lightgbm"
    )
    