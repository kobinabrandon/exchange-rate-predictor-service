import pickle 
from comet_ml import API
from src.logger import get_console_logger
from src.pipeline import Pipeline

def load_model_from_registry(
    workspace: str,
    api_key: str,
    model_name: str,
    status: str = "Production"
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
    model_details = api.get_registry_model_details(workspace, model_name)["versions"]
    
    # Search the dictionary to extract the versions of the models that are of the relevant status 
    model_versions = [
        detail["version"] for detail in model_details if detail["status"] == status
    ]
    
    if len(model_versions) == 0:
        
        logger.error("No production model found")
        raise ValueError("No production model found")
    
    else:
        logger.info(f"Found these {status} model versions: {model_versions}")
        model_version = model_versions[0]

    
    # Download the model from the registry and put it in a local file
    api.download_registry_model(
        workspace,
        registry_name=model_name,
        version=model_version,
        output_path="./",
        expand=True
    )
    
    
    with open("./model.pkl", "rb") as f:
        
        model = pickle.load(f)
        
    return model
        