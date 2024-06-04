import atexit
from datetime import datetime
import os
import subprocess
import shlex
import tempfile
from typing import Literal, Sequence

from tap import tapify


######################################## Utils #########################################


def run_command(command: str) -> str:
    result = subprocess.run(shlex.split(command), capture_output=True, text=True)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()
    return result.stdout


def write_temp_file(lines: Sequence[str]):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, "w") as file:
        file.writelines(lines)
    atexit.register(lambda: os.remove(temp_file.name))
    return temp_file


def concatenate_files(filenames: Sequence[str]):
    lines = []
    for file_idx, filename in enumerate(filenames):
        with open(filename, "r") as infile:
            lines.extend(infile.readlines())
            if file_idx != len(filenames) - 1:
                lines.append("\n\n")
    return write_temp_file(lines)


def write_default_cpu_test_experiment_file():
    default_experiment = "./experiment_mini.sh"
    lines = (
        'echo "Running experiment_mini.sh"\n',
        f"{default_experiment}\n",
    )
    print(f"Creating instance for running {default_experiment}")
    return write_temp_file(lines)


def write_default_gpu_experiment_file():
    default_experiment = "./experiment.sh"
    lines = (
        'echo "Running experiment.sh"\n',
        f"{default_experiment}\n",
    )
    print(f"Creating instance for running {default_experiment}")
    return write_temp_file(lines)


######################################### GCP ##########################################


create_instance_command_template_cpu_test = """
gcloud compute instances create {instance_name} \
    --project={project_name} \
    --zone={zone} \
    --machine-type=e2-standard-2 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --service-account={service_account_email} \
    --scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write \
    --create-disk=auto-delete=yes,boot=yes,device-name={instance_name},image=projects/debian-cloud/global/images/debian-12-bookworm-v20240515,mode=rw,size=80,type=projects/{project_name}/zones/{zone}/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \
    --metadata-from-file=startup-script={startup_script_name}
""".strip("\n")


create_instance_command_template_gpu = """
gcloud compute instances create {instance_name} \
    --project={project_name} \
    --zone={zone} \
    --machine-type=n1-highmem-2 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account={service_account_email} \
    --scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write \
    --accelerator=count=1,type=nvidia-tesla-t4 \
    --create-disk=auto-delete=yes,boot=yes,device-name={instance_name},image=projects/ml-images/global/images/c2-deeplearning-pytorch-2-2-cu121-v20240514-debian-11,mode=rw,size=80,type=projects/{project_name}/zones/{zone}/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \
    --metadata-from-file=startup-script={startup_script_name}
""".strip("\n")


def post_create_message(project_name: str, instance_name: str, zone: str) -> None:
    print(
        "View raw logs here (cleaner logs are in the Logs Explorer page):\n"
        f"https://console.cloud.google.com/compute/instancesDetail/zones/{zone}/instances/{instance_name}?project={project_name}&tab=monitoring&pageState=(%22timeRange%22:(%22duration%22:%22PT1H%22),%22observabilityTab%22:(%22mainContent%22:%22logs%22))"
    )
    print()
    print(
        "To SSH into the instance:\n"
        f'gcloud compute ssh --zone "{zone}" "{instance_name}" --project "{project_name}"'
    )
    print()


######################################### Main #########################################


def create_instance(
    cpu_test_or_gpu: Literal["cpu-test", "gpu"],
    project_name: str | None = None,
    instance_name_prefix: str = "instance-pretrain-on-test",
    zone: str | None = None,
    experiment_file_name: str | None = None,
):
    # Set up gcloud instance create arguments
    is_experiment_default = experiment_file_name is None
    if cpu_test_or_gpu == "cpu-test":
        create_instance_command_template = create_instance_command_template_cpu_test
        if is_experiment_default:
            experiment_file_name = write_default_cpu_test_experiment_file().name
        startup_script_filenames = (
            "./_preamble.sh",
            "../_setup_python_env.sh",
            experiment_file_name,
            "../_shutdown.sh",
        )
        if zone is None:
            zone = "us-central1-a"
    elif cpu_test_or_gpu == "gpu":
        create_instance_command_template = create_instance_command_template_gpu
        if is_experiment_default:
            experiment_file_name = write_default_gpu_experiment_file().name
        startup_script_filenames = (
            "./_install_cuda.sh",
            "./_preamble.sh",
            "../_setup_python_env.sh",
            experiment_file_name,
            "../_shutdown.sh",
        )
        if zone is None:
            zone = "us-west4-a"

    if project_name is None:
        project_name = run_command("gcloud config get-value project").strip("\n")
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    if is_experiment_default:
        instance_name_suffix = ""
    else:
        instance_name_suffix = "-" + os.path.basename(
            experiment_file_name
        ).removesuffix(".sh").replace("_", "-")
    instance_name = (
        f"{instance_name_prefix}-{cpu_test_or_gpu}-{current_time}{instance_name_suffix}"
    )

    # Create instance
    startup_script = concatenate_files(startup_script_filenames)
    create_instance_command = create_instance_command_template.format(
        project_name=project_name,
        instance_name=instance_name,
        zone=zone,
        service_account_email=os.environ["SERVICE_ACCOUNT_EMAIL"],
        startup_script_name=startup_script.name,
    )
    create_message = run_command(create_instance_command)
    print("\n" + create_message + "\n")
    post_create_message(project_name, instance_name, zone)


def create_instances(
    cpu_test_or_gpu: Literal["cpu-test", "gpu"],
    project_name: str | None = None,
    instance_name_prefix: str = "instance-pretrain-on-test",
    zone: str | None = None,
    experiment_dir: str | None = None,
):
    """
    Creates GPU instances which run the experiment files in `experiment_dir`—one
    instance per experiment.

    Parameters
    ----------
    cpu_test_or_gpu : Literal["cpu-test", "gpu"]
        Whether or not this is a CPU test or a full GPU run
    project_name : str | None, optional
        Name of the project for this experiment, by default `gcloud config get-value
        project`
    instance_name_prefix : str, optional
        Prefix of the name of the instance
    zone : str | None, optional
        Zone with CPU/GPU availability, by default us-central1-a for CPU and us-west4-a
        for GPU. Consider trying us-west4-b if the default fails.
    experiment_dir : str | None, optional
        Directory containing bash .sh files which will be run from the repo root by the
        cloud instance. Assume all cloud and Python env set up is complete. By default,
        ./experiment_mini.sh for CPU ./experiment.sh for GPU
    """
    if experiment_dir is None:
        sh_files = [None]
    else:
        experiment_file_names = sorted(os.listdir(experiment_dir))
        sh_files = [
            os.path.join(experiment_dir, filename)
            for filename in experiment_file_names
            if filename.endswith(".sh")
        ]
        if not sh_files:
            non_sh_files = "\n".join(experiment_file_names)
            raise ValueError(
                f"Expected bash .sh files in {experiment_dir}. Got:\n{non_sh_files}"
            )

    for experiment_file_name in sh_files:
        if experiment_file_name is not None:
            print(f"Creating instance for running {experiment_file_name}")
        create_instance(
            cpu_test_or_gpu,
            project_name=project_name,
            instance_name_prefix=instance_name_prefix,
            zone=zone,
            experiment_file_name=experiment_file_name,
        )
        print(("-" * os.get_terminal_size().columns) + "\n")


if __name__ == "__main__":
    tapify(create_instances)
