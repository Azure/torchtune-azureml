import json
import ipykernel
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data, Environment, BuildContext, Model
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

def check_kernel():

    kernel_id = ipykernel.connect.get_connection_file()

    with open(kernel_id, 'r') as f:
        data = json.load(f)  

    if data["kernel_name"] == "":
        print("Select kernel first!")
    else:
        print(f"Kernel: {data['kernel_name']}")
        
def get_or_create_environment_asset(ml_client, env_name, conda_yml="cloud/conda.yml", update=False):
    
    try:
        latest_env_version = max([int(e.version) for e in ml_client.environments.list(name=env_name)])
        if update:
            raise ResourceExistsError('Found Environment asset, but will update the Environment.')
        else:
            env_asset = ml_client.environments.get(name=env_name, version=latest_env_version)
            print(f"Found Environment asset: {env_name}. Will not create again")
    except (ResourceNotFoundError, ResourceExistsError) as e:
        print(f"Exception: {e}")        
        env_docker_image = Environment(
            image="mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest",
            conda_file=conda_yml,
            name=env_name,
            description="Environment created for llm fine-tuning.",
        )
        env_asset = ml_client.environments.create_or_update(env_docker_image)
        print(f"Created Environment asset: {env_name}")
        
    return env_asset


def get_or_create_docker_environment_asset(ml_client, env_name, docker_dir, update=False):
    
    try:
        latest_env_version = max([int(e.version) for e in ml_client.environments.list(name=env_name)])
        if update:
            raise ResourceExistsError('Found Environment asset, but will update the Environment.')
        else:
            env_asset = ml_client.environments.get(name=env_name, version=latest_env_version)
            print(f"Found Environment asset: {env_name}. Will not create again")
    except (ResourceNotFoundError, ResourceExistsError) as e:
        print(f"Exception: {e}")
        env_docker_image = Environment(
            build=BuildContext(path=docker_dir),
            name=env_name,
            description="Environment created from a Docker context.",
        )
        env_asset = ml_client.environments.create_or_update(env_docker_image)
        print(f"Created Environment asset: {env_name}")
    
    return env_asset


def get_or_create_data_asset(ml_client, data_name, data_local_dir, update=False):
    
    try:
        latest_data_version = max([int(d.version) for d in ml_client.data.list(name=data_name)])
        if update:
            raise ResourceExistsError('Found Data asset, but will update the Data.')            
        else:
            data_asset = ml_client.data.get(name=data_name, version=latest_data_version)
            print(f"Found Data asset: {data_name}. Will not create again")
    except (ResourceNotFoundError, ResourceExistsError) as e:
        data = Data(
            path=data_local_dir,
            type=AssetTypes.URI_FOLDER,
            description=f"{data_name} for fine tuning",
            tags={"FineTuningType": "Instruction", "Language": "En"},
            name=data_name
        )
        data_asset = ml_client.data.create_or_update(data)
        print(f"Created Data asset: {data_name}")
        
    return data_asset


def get_or_create_model_asset(ml_client, model_name, job_name, model_dir="outputs", model_type="custom_model", 
                              download_quantized_model_only=False, update=False):
    
    try:
        latest_model_version = max([int(m.version) for m in ml_client.models.list(name=model_name)])
        if update:
            raise ResourceExistsError('Found Model asset, but will update the Model.')
        else:
            model_asset = ml_client.models.get(name=model_name, version=latest_model_version)
            print(f"Found Model asset: {model_name}. Will not create again")
    except (ResourceNotFoundError, ResourceExistsError) as e:
        print(f"Exception: {e}")
        model_path = f"azureml://jobs/{job_name}/outputs/artifacts/paths/{model_dir}"    
        if download_quantized_model_only:
            model_path = f"azureml://jobs/{job_name}/outputs/artifacts/paths/{model_dir}/quant"    
        run_model = Model(
            name=model_name,        
            path=model_path,
            description="Model created from run.",
            type=model_type # mlflow_model, custom_model, triton_model
        )
        model_asset = ml_client.models.create_or_update(run_model)
        print(f"Created Model asset: {model_name}")

    return model_asset


def get_num_gpus(azure_compute_cluster_size):
    num_gpu_dict = {
        "Standard_NC24ads_A100_v4": 1,
        "Standard_NC48ads_A100_v4": 2,
        "Standard_NC96ads_A100_v4": 4,
        "Standard_NC40ads_H100_v5": 1,
        "Standard_NC80adis_H100_v5": 2    
    }
    return num_gpu_dict[azure_compute_cluster_size]


