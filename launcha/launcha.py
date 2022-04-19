import argparse
import os
import shutil
import subprocess
import time
from distutils.util import strtobool

import boto3
import requests
import wandb

import launcha

# by default we suggest using 80% of the available memory
# when submitting the job (otherwise it might fails in our experience)
MEMORY_USAGE_FRACTION = 0.8


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Launcha CLI')
    subparsers = parser.add_subparsers(dest='subcommand')
    init = subparsers.add_parser('init', help='initialize the terraform template in the current folder')
    
    parser.add_argument('-d', '--docker-tag', type=str, default="vwxyzjn/cleanrl:latest",
        help='the name of the docker tag')
    parser.add_argument('--command', type=str, default="poetry run python cleanrl/ppo.py",
        help='the docker command')

    # Wandb args for experiment management
    parser.add_argument('--wandb-key', type=str, default="",
        help='the wandb key. If not provided, the script will try to read from `netrc`')

    # AWS Batch args for experiment submission
    parser.add_argument('--instance-type', type=str, default="g4dn.xlarge",
        help='the instance type name (e.g., g4dn.xlarge)')
    parser.add_argument('--num-vcpu', type=int, default=0,
        help='number of vcpu per experiment')
    parser.add_argument('--num-memory', type=int, default=0,
        help='number of memory (MB) per experiment')
    parser.add_argument('--num-gpu', type=int, default=0,
        help='number of gpu per experiment')
    parser.add_argument('--num-hours', type=float, default=120.0,
        help='number of hours allocated experiment')
    parser.add_argument('-b', '--build', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the script will build a container')
    parser.add_argument('--archs', type=str, default="linux/amd64", # linux/arm64,linux/amd64
        help='the archs to build the docker container for')
    parser.add_argument('-p', '--push', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the script will push the built container')
    parser.add_argument('--provider', type=str, default="", choices=["aws"],
        help='the cloud provider of choice (currently only `aws` is supported)')
    parser.add_argument('--aws-num-retries', type=int, default=1,
        help='the number of job retries for `provider=="aws"`')
    args = parser.parse_args()
    # fmt: on
    return args


def recommend_specs(instance_type: str):
    ec2_client = boto3.client("ec2")
    specs = ec2_client.describe_instance_types(InstanceTypes=[instance_type])["InstanceTypes"][0]
    num_vcpu = specs["VCpuInfo"]["DefaultVCpus"]
    num_memory = int(specs["MemoryInfo"]["SizeInMiB"] * MEMORY_USAGE_FRACTION)
    gpu_count = 0
    if "GpuInfo" in specs:
        gpus_specs = specs["GpuInfo"]["Gpus"]
        for gpus_spec in gpus_specs:
            gpu_count += gpus_spec["Count"]
    print(f"Recommended usage for {instance_type}: {num_vcpu} vcpu, {num_memory} MB memory, {gpu_count} gpu")
    return num_vcpu, num_memory, gpu_count


def submit_aws_job(
    docker_tag: str,
    run_command: str,
    instance_type: str,
    num_vcpu: int = 0,
    num_memory: int = 0,
    num_gpu: int = 0,
    num_hours: float = 120.0,
    num_retries: int = 1,
    environment_variables: dict[str, str] = []
):
    recommended_num_vcpu, recommended_num_memory, recommended_gpu_count = recommend_specs(instance_type)
    if num_vcpu == 0:
        num_vcpu = recommended_num_vcpu
    if num_memory == 0:
        num_memory = recommended_num_memory
    if num_gpu == 0:
        num_gpu = recommended_gpu_count

    batch_client = boto3.client("batch")
    job_name = docker_tag.replace(":", "").replace("/", "_").replace(" ", "").replace("-", "_") + str(int(time.time()))
    resources_requirements = []
    if num_gpu:
        resources_requirements = [
            {"value": str(num_gpu), "type": "GPU"},
        ]
    try:
        job_def_name = docker_tag.replace(":", "_").replace("/", "_")
        batch_client.register_job_definition(
            jobDefinitionName=job_def_name,
            type="container",
            containerProperties={
                "image": docker_tag,
                "vcpus": num_vcpu,
                "memory": num_memory,
                "command": [
                    "/bin/bash",
                ],
            },
        )
        response = batch_client.submit_job(
            jobName=job_name,
            jobQueue=instance_type,
            jobDefinition=job_def_name,
            containerOverrides={
                "vcpus": num_vcpu,
                "memory": num_memory,
                "command": ["/bin/bash", "-c", run_command],
                "environment": environment_variables,
                "resourceRequirements": resources_requirements,
            },
            retryStrategy={"attempts": num_retries},
            timeout={"attemptDurationSeconds": int(num_hours * 60 * 60)},
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            print(response)
            raise Exception("jobs submit failure")
    except Exception as e:
        print(e)
    finally:
        response = batch_client.deregister_job_definition(jobDefinition=job_def_name)
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            print(response)
            raise Exception("jobs submit failure")


def build_docker(push: bool, archs: str, docker_tag: str):
    output_type_str = "--output=type=registry" if push else "--output=type=docker"
    subprocess.run(
        f"docker buildx build {output_type_str} --platform {archs} -t {docker_tag} .",
        shell=True,
        check=True,
    )


def main():
    args = parse_args()
    if args.subcommand == "init":
        shutil.copytree(os.path.join(launcha.__path__[0], "template"), ".", dirs_exist_ok=True)
        print(
            """
Terraform template files initialized. Spin up the AWS computing environments by running:
`terraform init`
`terraform apply`
The computing environments' setup is free of charge. You will only be billed when you submit jobs.
        """
        )
        return

    if args.build:
        build_docker(args.push, args.archs, args.docker_tag)

    if not args.wandb_key:
        try:
            args.wandb_key = requests.utils.get_netrc_auth("https://api.wandb.ai")[-1]
        except:
            pass
    assert len(args.wandb_key) > 0, "you have not logged into W&B; try do `wandb login`"


    run_command = (
        f"docker run -d -e WANDB_API_KEY={args.wandb_key} {args.docker_tag} "
        + '/bin/bash -c "'
        + args.command
        + '"'
        + "\n"
    )
    print(run_command)

    # submit jobs
    if args.provider == "aws":
        submit_aws_job(
            args.docker_tag,
            run_command,
            args.instance_type,
            num_vcpu=args.num_vcpu,
            num_memory=args.num_memory,
            num_gpu=args.num_gpu,
            num_hours=args.num_hours,
            num_retries=args.aws_num_retries,
            environment_variables=[
                {"name": "WANDB_API_KEY", "value": args.wandb_key},
                {"name": "WANDB_RESUME", "value": "allow"},
                {"name": "WANDB_RUN_ID", "value": wandb.util.generate_id()},
            ]
        )

