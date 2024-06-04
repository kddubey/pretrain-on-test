import contextlib
from datetime import datetime
import os
import subprocess
import shlex
import shutil
from typing import Literal, Sequence

from tap import tapify


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


def run_command(command: str) -> str:
    result = subprocess.run(shlex.split(command), capture_output=True, text=True)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()
    return result.stdout


def remove_file_if_exists(filename: str):
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


def concatenate_files(
    infilenames: Sequence[str], outfilename: str = "startup_script.sh"
) -> str:
    """
    ```bash
    cat *infilenames > outfilename
    ```
    """
    remove_file_if_exists(outfilename)
    # https://stackoverflow.com/a/27077437/18758987
    with open(outfilename, "wb") as outfile:
        for filename in infilenames:
            with open(filename, "rb") as infile:
                shutil.copyfileobj(infile, outfile)
    return outfilename


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


def create_instance(
    cpu_test_or_gpu: Literal["cpu-test", "gpu"],
    project_name: str | None = None,
    instance_name_prefix: str = "instance-pretrain-on-test",
    zone: str | None = None,
):
    """
    Creates a GPU instance which runs the experiment at the bottom of ../run.sh.

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
        for GPU
    """
    if cpu_test_or_gpu == "cpu-test":
        create_instance_command_template = create_instance_command_template_cpu_test
        startup_script_filenames = ("./_preamble.sh", "../run.sh")
        zone = zone or "us-central1-a"
    elif cpu_test_or_gpu == "gpu":
        create_instance_command_template = create_instance_command_template_gpu
        startup_script_filenames = ("./_install_cuda.sh", "./_preamble.sh", "../run.sh")
        zone = zone or "us-west4-a"

    # Set up gcloud instance create arguments
    if project_name is None:
        project_name = run_command("gcloud config get-value project").strip("\n")
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    instance_name = f"{instance_name_prefix}-{cpu_test_or_gpu}-{current_time}"

    startup_script_name = concatenate_files(infilenames=startup_script_filenames)
    try:
        create_instance_command = create_instance_command_template.format(
            project_name=project_name,
            instance_name=instance_name,
            zone=zone,
            service_account_email=os.environ["SERVICE_ACCOUNT_EMAIL"],
            startup_script_name=startup_script_name,
        )
        create_message = run_command(create_instance_command)
        print("\n" + create_message + "\n")
        post_create_message(project_name, instance_name, zone)
    finally:
        remove_file_if_exists(startup_script_name)


if __name__ == "__main__":
    tapify(create_instance)
