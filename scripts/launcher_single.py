import os
import sys
import json
import time
import argparse
import socket
import wandb
import jinja2
import torch
import subprocess as sb
from pathlib import Path
from typing import Dict, Optional, Tuple

def download_model(model_id:str, model_dir: str, ignore_patterns: str="") -> None:
    """
    Download a model if necessary.

    Args:
        model_output_folder (str): The folder to store the downloaded model.
        args (argparse.Namespace): Command-line arguments.
    """
    if ignore_patterns == "":
        full_command = f'tune download {model_id} --output-dir {model_dir} --hf-token {args.hf_token} --ignore-patterns None'
    else:
        full_command = f'tune download {model_id} --output-dir {model_dir} --hf-token {args.hf_token} --ignore-patterns "{ignore_patterns}"'
        
    
    if not args.use_downloaded_model:
        print("Downloading model...")
        #delete_model_artifacts=f'rm -rf {model_dir}/*'
        #run_command(delete_model_artifacts)
        
        list_models=f'ls -ltr {model_dir}'
        run_command(list_models)

        run_command(full_command)
    else:
        print("Using existing downloaded model.")


def set_custom_env(env_vars: Dict[str, str]) -> None:
    """
    Set custom environment variables.

    Args:
        env_vars (Dict[str, str]): A dictionary of environment variables to set.
                                   Keys are variable names, values are their corresponding values.

    Returns:
        None

    Raises:
        TypeError: If env_vars is not a dictionary.
        ValueError: If any key or value in env_vars is not a string.
    """
    if not isinstance(env_vars, dict):
        raise TypeError("env_vars must be a dictionary")

    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All keys and values in env_vars must be strings")

    os.environ.update(env_vars)

    # Optionally, print the updated environment variables
    print("Updated environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")


def finetune_model() -> None:
    """
    Fine-tune a model
    
    Returns:
        None
    """
    print("***** Starting model fine-tuning *****")
    
    # Set custom environment variables
    # NCCL_DEBUG=INFO will dump a lot of NCCL-related debug information, which you can then search online if you find that some problems are reported.
    # Or if you’re not sure how to interpret the output you can share the log file in an Issue.
    custom_env: Dict[str, str] = {
        "HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
        "HF_TOKEN": args.hf_token, 
        #"NCCL_DEBUG": "INFO",
        "WANDB_API_KEY": args.wandb_api_key,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_WATCH": args.wandb_watch,        
        "WANDB_DIR": args.log_dir
    }
    
    set_custom_env(custom_env) 
    os.makedirs(args.model_dir, exist_ok=True)    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_output_dir, exist_ok=True)
    
    # Download the model
    download_model(args.model_id, args.model_dir)
    
    # Construct the fine-tuning command
    print("***** Single Device Training *****");
    full_command = (
        f'tune run '
        f'{args.tune_recipe} '
        f'--config {args.tune_config_name}'
    )
    # Run the fine-tuning command
    run_command(full_command)


def run_eval() -> None:
    """
    Run evaluation on the model.

    This function sets up the environment, downloads the model,
    and runs the evaluation command.

    Args:
        args: An object containing command-line arguments.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If any subprocess command fails.
    """
    print("***** Starting model evaluation *****")

    # Construct the evaluation command
    full_command = f'tune run eleuther_eval --config evaluation.yaml'
    
    print("Running evaluation command...")
    run_command(full_command)

    
def run_quant() -> None:
    """
    Run quantization on the model.

    This function sets up the environment, displays the configuration,
    and runs the quantization command if it's on the primary node.

    Args:
        args: An object containing command-line arguments.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If any subprocess command fails.
    """
    print("***** Starting model quantization *****")
    
    # Construct the quantization command
    full_command = f'tune run quantize --config quant.yaml'

    print("Running quantization on primary node...")
    run_command(full_command)

        
def run_command(command: str) -> None:
    """
    Run a shell command and handle potential errors.

    Args:
        command (str): The command to run.

    Raises:
        subprocess.CalledProcessError: If the command fails.
        ValueError: If the command string is empty.
        
    """

    print(f'\n\n ***** Executing command: {command} \n\n')

    try:
        # Start the timer
        start_time = time.time()
        
        result = sb.run(
            command,
            shell=True,
            capture_output=False,
            text=True,
            check=True
        )
        # End the timer
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        print(f"\n\n ***** Execution time for command: {command} is : {elapsed_time:.4f} seconds \n\n")

    except sb.CalledProcessError as e:
        report_error=1
        print(f"**** Command failed with error code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        raise
    except Exception as e:
        report_error=1
        print(f"****An unexpected error occurred: {e}")
        raise   
        
    
def check_pytorch_version() -> Optional[str]:
    """
    Check and return the installed PyTorch version.

    This function runs a Python command to import torch and print its version.

    Returns:
        Optional[str]: The PyTorch version as a string, or None if an error occurred.

    Raises:
        subprocess.CalledProcessError: If the subprocess command fails.
    """
    try:
        # Run the command to get the PyTorch version
        result = sb.run(
            ['python', '-c', 'import torch; print(torch.__version__)'],
            capture_output=True,
            text=True,
            check=True
        )

        # Extract and strip the version string
        version = result.stdout.strip()

        print(f"Installed PyTorch version: {version}")
        return version

    except sb.CalledProcessError as e:
        print(f"Error occurred while checking PyTorch version: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return None
    

def parse_arge():

    parser = argparse.ArgumentParser()

    # infra configuration
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--train_dir", type=str, default="train")    
    parser.add_argument("--model_dir", type=str, default="../model")
    parser.add_argument("--log_dir", type=str, default="../log")
    parser.add_argument("--model_output_dir", type=str, default="../output") 
    parser.add_argument("--tune_config_name", type=str, default="lora_finetune.yaml")
    parser.add_argument("--prompt", type=str, default="") 
    parser.add_argument("--hf_token", type=str, default="") 
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_watch", type=str, default="gradients") # options: false | gradients | all    
    parser.add_argument("--tune_recipe", type=str, default="lora_finetune_single_device")     
    parser.add_argument("--tune_action", type=str, default="fine-tune")
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument('--use_downloaded_model', type=bool, default=False)

    args = parser.parse_known_args()
    
    return args

def print_env_vars():

    print("***** Printing enviroment variables *****")
    print(f"Use Downloaded Model: {args.use_downloaded_model}")
    print(f"Type of use_downloaded_model: {type(args.use_downloaded_model)}")
    print(f"Action: {args.tune_action}")
    
    check_pytorch_version()
    
def completion_status():
    print("***** Finished Task *****")
        
    list_model_dir=f'ls -ltr {args.model_dir}'
    run_command(list_model_dir)
        
    list_quantized_model_dir = f'ls -ltr {args.model_dir}/quantized'
    run_command(list_quantized_model_dir)
    
def training_function():
    
    print_env_vars()

    # Step 1: Map values to functions
    function_map = {
        "fine-tune": finetune_model,
        "run-eval": run_eval,
        "run-quant": run_quant
    }
    
    # Step 2: Iterate through the array and call the corresponding functions
    for value in args.tune_action.split(","):
        if value in function_map:
            print(f'function_key: {value}')
            try:
                function_map[value]()
            except Exception as e:
                print(f"An error occurred in function {value}: {e}")
                raise e
        else:
            print(f"No function defined for value {value}")


if __name__ == "__main__":
    
    report_error = 0
    args, _ = parse_arge()
    print(args)

    # get the current working directory
    current_working_directory = os.getcwd()

    # print output to the console
    print(current_working_directory)

    jinja_env = jinja2.Environment()  
    
    # Dynamically modify fine-tuning yaml file.
    template = jinja_env.from_string(Path(args.tune_config_name).open().read())
    train_path = os.path.join(args.train_dir, "train.jsonl");
    metric_logger = "DiskLogger"
    if len(args.wandb_api_key) > 0:
        metric_logger = "WandBLogger"
        
    Path(args.tune_config_name).open("w").write(
        template.render(
            train_path=train_path, 
            log_dir=args.log_dir, 
            model_dir=args.model_dir, 
            model_output_dir=args.model_output_dir,
            metric_logger=metric_logger
        )
    )
    
    # Dynamically modify Evaluation yaml file.
    template = jinja_env.from_string(Path("evaluation.yaml").open().read())
    Path("evaluation.yaml").open("w").write(
        template.render(
            model_dir=args.model_dir, 
            model_output_dir=args.model_output_dir
        )
    )
    
    # Dynamically modify Quantization yaml file.
    template = jinja_env.from_string(Path("quant.yaml").open().read())
    Path("quant.yaml").open("w").write(
        template.render(
            model_output_dir=args.model_output_dir
        )
    )
    
    #num_of_hosts,default_node_rank,leader, current_host = get_host_details()

    try:
        print("Starting training...")
        training_function()
        
        if(report_error==1):
            sys.exit(1)
            
        print(f"Training completed with code: {report_error}")

    except Exception as e:
        # Log the error
        print(f"Error occurred during training: {str(e)}")

        # Exit with a non-zero status code
        sys.exit(1)
    