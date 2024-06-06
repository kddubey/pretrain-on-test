import atexit
from datetime import datetime
import os
import subprocess
import shlex
import tempfile
from typing import Sequence

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


def _write_default_experiment_file(filename: str):
    lines = (
        f'echo "Running {filename}"\n',
        f"{filename}\n",
    )
    print(f"Creating instance for running {filename}")
    return write_temp_file(lines)


def write_default_experiment_file_cpu_test():
    return _write_default_experiment_file("./experiment_mini.sh")


def write_default_experiment_file_gpu():
    return _write_default_experiment_file("./experiment.sh")


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
    --service-account={gcp_service_account_email} \
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
    --service-account={gcp_service_account_email} \
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


def service_account_email(
    display_name: str = "Compute Engine default service account",
) -> str:
    exception = ValueError(
        "Couldn't find your GCP service account. Please input it as an argument: "
        "--gcp_service_account_email "
        '"xxxxxxxxxxxx-compute@developer.gserviceaccount.com"'
    )
    try:
        email = run_command(
            "gcloud iam service-accounts list "
            f'--filter="displayName:{display_name}" '
            '--format="value(email)"'
        ).strip(" \n")
    except subprocess.CalledProcessError:  # missing perms or wrong display name
        raise exception
    else:
        if email and "@" in email:
            return email
        else:
            raise exception


def post_create_message(
    project_name: str, instance_name: str, zone: str, dont_run_experiment: bool
) -> None:
    print(
        "View raw logs here (cleaner logs are in the Logs Explorer page):\n"
        f"https://console.cloud.google.com/compute/instancesDetail/zones/{zone}"
        f"/instances/{instance_name}?project={project_name}&tab=monitoring&pageState="
        f"(%22timeRange%22:(%22duration%22:%22PT1H%22),%22observabilityTab%22:(%22main"
        "Content%22:%22logs%22))"
    )
    print()
    if dont_run_experiment:
        print(
            "To SSH into the instance, wait 1-2 minutes and then (in a new terminal "
            "tab) run:\n"
            f'gcloud compute ssh --zone "{zone}" "{instance_name}" --project '
            f'"{project_name}"'
        )
        print()


######################################### Main #########################################


def create_instance(
    experiment_file_name: str | None = None,
    zone: str | None = None,
    is_cpu_test: bool = False,
    project_name: str | None = None,
    instance_name_prefix: str = "instance-pretrain-on-test",
    gcp_service_account_email: str | None = None,
    dont_run_experiment: bool = False,
):
    # Set up gcloud instance create arguments
    is_experiment_default = experiment_file_name is None
    if is_cpu_test:
        create_instance_command_template = create_instance_command_template_cpu_test
        if is_experiment_default:
            experiment_file_name = write_default_experiment_file_cpu_test().name
        startup_script_filenames = (
            "./_preamble.sh",
            "../_setup_python_env.sh",
            experiment_file_name,
            "../_shutdown.sh",
        )
        if zone is None:
            zone = "us-central1-a"
    else:
        create_instance_command_template = create_instance_command_template_gpu
        if is_experiment_default:
            experiment_file_name = write_default_experiment_file_gpu().name
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
    instance_name = "-".join(
        (
            instance_name_prefix,
            "cpu-test" if is_cpu_test else "gpu",
            current_time,
            ""
            if is_experiment_default
            else os.path.basename(experiment_file_name)
            .removesuffix(".sh")
            .replace("_", "-"),
        )
    ).rstrip("-")
    if gcp_service_account_email is None:
        gcp_service_account_email = service_account_email()
    if dont_run_experiment:
        startup_script_filenames = ("./_preamble.sh",)

    # Create instance
    startup_script = concatenate_files(startup_script_filenames)
    create_instance_command = create_instance_command_template.format(
        project_name=project_name,
        instance_name=instance_name,
        zone=zone,
        gcp_service_account_email=gcp_service_account_email,
        startup_script_name=startup_script.name,
    )
    create_message = run_command(create_instance_command)
    print("\n" + create_message + "\n")
    post_create_message(project_name, instance_name, zone, dont_run_experiment)


def create_instances(
    experiment_dir: str | None = None,
    zone: str | None = None,
    is_cpu_test: bool = False,
    project_name: str | None = None,
    instance_name_prefix: str = "instance-pretrain-on-test",
    gcp_service_account_email: str | None = None,
    dont_run_experiment: bool = False,
):
    """
    Creates instances which run the experiment files in `experiment_dir`, where one
    instance corresponds to one experiment file.

    Parameters
    ----------
    experiment_dir : str | None, optional
        Directory containing bash .sh files which will be run from the repo root by the
        cloud instance. Assume all cloud and Python env set up is complete. By default
        ./experiment_mini.sh for CPU ./experiment.sh for GPU.
    zone : str | None, optional
        Zone with CPU/GPU availability, by default us-central1-a for CPU and us-west4-a
        for GPU. Consider trying us-west4-b if the default fails.
    is_cpu_test : bool, optional
        Whether or not this is a CPU test or a full GPU run. By default, it's a full GPU
        run.
    project_name : str | None, optional
        Name of the project for this experiment, by default `gcloud config get-value
        project`.
    instance_name_prefix : str, optional
        Prefix of the name of the instance.
    gcp_service_account_email : str | None, optional
        Your Google Cloud service account. By default it's the one with display name
        "Compute Engine default service account".
    dont_run_experiment : bool, optional
        Just set up the instance for SSHing into instead of running the experiment. By
        default, the instance will run the experiment.
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
            experiment_file_name=experiment_file_name,
            zone=zone,
            is_cpu_test=is_cpu_test,
            project_name=project_name,
            instance_name_prefix=instance_name_prefix,
            gcp_service_account_email=gcp_service_account_email,
            dont_run_experiment=dont_run_experiment,
        )
        print(("-" * os.get_terminal_size().columns) + "\n")


if __name__ == "__main__":
    tapify(create_instances)
