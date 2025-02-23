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
from contextlib import contextmanager
import torch.distributed as dist
from util import get_epoch


MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "7777")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
GLOBAL_RANK = int(os.environ.get("RANK", -1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))

NUM_GPUS_PER_NODE = torch.cuda.device_count()
NUM_NODES = WORLD_SIZE // NUM_GPUS_PER_NODE

if LOCAL_RANK != -1:
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training
    wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def download_model(model_id: str, model_dir: str, ignore_patterns: str = "") -> None:
    """
    Download a model if necessary.

    Args:
        model_output_folder (str): The folder to store the downloaded model.
        args (argparse.Namespace): Command-line arguments.
    """
    if ignore_patterns == "":
        full_command = f"tune download {model_id} --output-dir {model_dir} --hf-token {args.hf_token} --ignore-patterns None"
    else:
        full_command = f'tune download {model_id} --output-dir {model_dir} --hf-token {args.hf_token} --ignore-patterns "{ignore_patterns}"'

    if not args.use_downloaded_model:
        print("Downloading model...")
        # delete_model_artifacts=f'rm -rf {model_dir}/*'
        # run_command(delete_model_artifacts)

        list_models = f"ls -ltr {model_dir}"
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
    Fine-tune a model using distributed training.

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
        # "NCCL_DEBUG": "INFO",
        "WANDB_API_KEY": args.wandb_api_key,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_WATCH": args.wandb_watch,
        "WANDB_DIR": args.log_dir,
    }

    set_custom_env(custom_env)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_output_dir, exist_ok=True)

    with torch_distributed_zero_first(LOCAL_RANK):
        # Download the model
        download_model(args.model_id, args.model_dir, args.ignore_patterns)

    # Construct the fine-tuning command
    if "single" in args.tune_recipe:
        print("***** Single Device Training *****")
        full_command = f"tune run {args.tune_recipe} --config {args.tune_finetune_yaml}"
        # Run the fine-tuning command
        run_command(full_command)
    else:
        print("***** Distributed Training *****")
        if dist.is_initialized():
            print("Destroying current process group before launching tune run...")
            dist.destroy_process_group()

        if GLOBAL_RANK in {-1, 0}:
            # Run the fine-tuning command
            full_command = (
                f"tune run --master-addr {MASTER_ADDR} --master-port {MASTER_PORT} --nnodes {NUM_NODES} --nproc_per_node {NUM_GPUS_PER_NODE} "
                f"{args.tune_recipe} "
                f"--config {args.tune_finetune_yaml}"
            )
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

    if LOCAL_RANK != -1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Set custom environment variables
    custom_env: Dict[str, str] = {
        "HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
        "HF_TOKEN": args.hf_token,
    }
    set_custom_env(custom_env)

    # Construct the evaluation command
    full_command = f"tune run eleuther_eval --config {args.tune_eval_yaml}"

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
    full_command = f"tune run quantize --config {args.tune_quant_yaml}"

    if GLOBAL_RANK in {-1, 0}:
        print("Running quantization on primary node...")
        run_command(full_command)
    else:
        print("Not on primary node. Skipping quantization.")


def run_command(command: str) -> None:
    """
    Run a shell command and handle potential errors.

    Args:
        command (str): The command to run.

    Raises:
        subprocess.CalledProcessError: If the command fails.
        ValueError: If the command string is empty.

    """

    print(f"\n\n ***** Executing command: {command} \n\n")

    try:
        # Start the timer
        start_time = time.time()

        result = sb.run(
            command, shell=True, capture_output=False, text=True, check=True
        )
        # End the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(
            f"\n\n ***** Execution time for command: {command} is : {elapsed_time:.4f} seconds \n\n"
        )

    except sb.CalledProcessError as e:
        report_error = 1
        print(f"**** Command failed with error code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        raise
    except Exception as e:
        report_error = 1
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
            ["python", "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            check=True,
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
    parser.add_argument("--ignore_patterns", type=str, default="")
    parser.add_argument(
        "--tune_finetune_yaml", type=str, default="lora_finetune_phi3.yaml"
    )
    parser.add_argument("--tune_eval_yaml", type=str, default="evaluation_phi3.yaml")
    parser.add_argument("--tune_quant_yaml", type=str, default="quant_phi3.yaml")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument(
        "--wandb_watch", type=str, default="gradients"
    )  # options: false | gradients | all
    parser.add_argument(
        "--tune_recipe", type=str, default="lora_finetune_single_device"
    )
    parser.add_argument("--tune_action", type=str, default="fine-tune")
    parser.add_argument(
        "--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct"
    )
    parser.add_argument("--use_downloaded_model", type=bool, default=False)

    args = parser.parse_known_args()

    return args


def print_env_vars():

    print("***** Printing enviroment variables *****")
    print(f"Master Addr: {MASTER_ADDR}")
    print(f"Mater Port: {MASTER_PORT}")
    print(f"Total number of GPUs (WORLD SIZE): {WORLD_SIZE}")
    print(f"The (global) rank of the current process: {GLOBAL_RANK}")
    print(f"Local node rank: {LOCAL_RANK}")
    print(f"Number of GPUs per node: {NUM_GPUS_PER_NODE}")
    print(f"Number of nodes: {NUM_NODES}")
    print(f"Use Downloaded Model: {args.use_downloaded_model}")
    print(f"Type of use_downloaded_model: {type(args.use_downloaded_model)}")
    print(f"Action: {args.tune_action}")
    check_pytorch_version()


def completion_status():
    print("***** Finished Task *****")

    list_model_dir = f"ls -ltr {args.model_dir}"
    run_command(list_model_dir)

    list_quantized_model_dir = f"ls -ltr {args.model_dir}/quantized"
    run_command(list_quantized_model_dir)


def training_function():

    print_env_vars()

    # Step 1: Map values to functions
    function_map = {
        "fine-tune": finetune_model,
        "run-eval": run_eval,
        "run-quant": run_quant,
    }

    # Step 2: Iterate through the array and call the corresponding functions
    for value in args.tune_action.split(","):
        if value in function_map:
            print(f"function_key: {value}")
            try:
                if value != "fine-tune" and dist.is_initialized():
                    print(
                        "Destroying current process group before executing the next action..."
                    )
                    dist.destroy_process_group()
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
    template = jinja_env.from_string(Path(args.tune_finetune_yaml).open().read())
    train_path = os.path.join(args.train_dir, "train.jsonl")
    metric_logger = "DiskLogger"
    if len(args.wandb_api_key) > 0:
        metric_logger = "WandBLogger"

    Path(args.tune_finetune_yaml).open("w").write(
        template.render(
            train_path=train_path,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            model_output_dir=args.model_output_dir,
            metric_logger=metric_logger,
        )
    )

    epoch = get_epoch(args.tune_finetune_yaml)

    # Dynamically modify Evaluation yaml file.
    template = jinja_env.from_string(Path(args.tune_eval_yaml).open().read())
    Path(args.tune_eval_yaml).open("w").write(
        template.render(
            model_dir=args.model_dir,
            model_output_dir=os.path.join(args.model_output_dir, f"epoch_{epoch}"),
        )
    )

    # Dynamically modify Quantization yaml file.
    template = jinja_env.from_string(Path(args.tune_quant_yaml).open().read())
    Path(args.tune_quant_yaml).open("w").write(
        template.render(
            model_output_dir=os.path.join(args.model_output_dir, f"epoch_{epoch}")
        )
    )

    try:
        print("Starting training...")
        training_function()

        if report_error == 1:
            sys.exit(1)

        print(f"Training completed with code: {report_error}")

    except Exception as e:
        # Log the error
        print(f"Error occurred during training: {str(e)}")

        # Exit with a non-zero status code
        sys.exit(1)

    if dist.is_initialized():
        dist.destroy_process_group()
