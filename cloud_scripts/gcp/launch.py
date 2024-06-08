import atexit
from datetime import datetime
import os
import re
import shlex
import subprocess
import tempfile
from time import sleep
from typing import Callable, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict
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


def write_experiment_mini():
    return _write_default_experiment_file("./experiment_mini.sh")


def write_experiment_full():
    return _write_default_experiment_file("./experiment.sh")


######################################### GCP ##########################################


def create_instance_command_cpu(
    instance_name: str,
    project_name: str,
    zone: str,
    gcp_service_account_email: str,
    startup_script_filename: str,
):
    return (
        f"gcloud compute instances create {instance_name} "
        f"--project={project_name} "
        f"--zone={zone} "
        f"--machine-type=e2-standard-2 "
        f"--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default "
        f"--no-restart-on-failure "
        f"--maintenance-policy=TERMINATE "
        f"--provisioning-model=SPOT "
        f"--instance-termination-action=DELETE "
        f"--service-account={gcp_service_account_email} "
        f"--scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write "
        f"--create-disk=auto-delete=yes,boot=yes,device-name={instance_name},image=projects/debian-cloud/global/images/debian-12-bookworm-v20240515,mode=rw,size=80,type=projects/{project_name}/zones/{zone}/diskTypes/pd-balanced "
        f"--no-shielded-secure-boot "
        f"--shielded-vtpm "
        f"--shielded-integrity-monitoring "
        f"--labels=goog-ec-src=vm_add-gcloud "
        f"--reservation-affinity=any "
        f"--metadata-from-file=startup-script={startup_script_filename}"
    )


def create_instance_command_gpu(
    instance_name: str,
    project_name: str,
    zone: str,
    gcp_service_account_email: str,
    startup_script_filename: str,
):
    return (
        f"gcloud compute instances create {instance_name} "
        f"--project={project_name} "
        f"--zone={zone} "
        f"--machine-type=n1-highmem-2 "
        f"--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default "
        f"--maintenance-policy=TERMINATE "
        f"--provisioning-model=STANDARD "
        f"--service-account={gcp_service_account_email} "
        f"--scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write "
        f"--accelerator=count=1,type=nvidia-tesla-t4 "
        f"--create-disk=auto-delete=yes,boot=yes,device-name={instance_name},image=projects/ml-images/global/images/c2-deeplearning-pytorch-2-2-cu121-v20240514-debian-11,mode=rw,size=80,type=projects/{project_name}/zones/{zone}/diskTypes/pd-balanced "
        f"--no-shielded-secure-boot "
        f"--shielded-vtpm "
        f"--shielded-integrity-monitoring "
        f"--labels=goog-ec-src=vm_add-gcloud "
        f"--reservation-affinity=any "
        f"--metadata-from-file=startup-script={startup_script_filename}"
    )


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


def create_startup_script_filenames(experiment_file_name: str):
    return (
        "./_preamble.sh",
        "../_setup_python_env.sh",
        experiment_file_name,
        "../_shutdown.sh",
    )


@runtime_checkable
class CreateInstanceCommand(Protocol):
    def __call__(
        self,
        instance_name: str,
        project_name: str,
        zone: str,
        gcp_service_account_email: str,
        startup_script_filename: str,
    ) -> str:
        """
        Returns a `gcloud compute instances create` command
        """


class ExperimentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    # Pydantic stuff: extra attributes are not allowed, and the object is immutable

    create_instance_command: CreateInstanceCommand
    write_default_experiment_file: Callable[[], tempfile._TemporaryFileWrapper]
    default_zone: str
    create_startup_script_filenames: Callable[[str], tuple[str]]


ExperimentTypes = Literal["cpu-test", "gpu", "gpu-test"]


experiment_type_to_info: dict[ExperimentTypes, ExperimentInfo] = {
    "cpu-test": ExperimentInfo(
        create_instance_command=create_instance_command_cpu,
        write_default_experiment_file=write_experiment_mini,
        default_zone="us-central1-a",
        create_startup_script_filenames=create_startup_script_filenames,
    ),
    "gpu": ExperimentInfo(
        create_instance_command=create_instance_command_gpu,
        write_default_experiment_file=write_experiment_full,
        default_zone="us-west4-a",
        create_startup_script_filenames=lambda experiment_file_name: (
            "./_install_cuda.sh",
        )
        + create_startup_script_filenames(experiment_file_name),
    ),
    "gpu-test": ExperimentInfo(
        create_instance_command=create_instance_command_gpu,
        write_default_experiment_file=write_experiment_mini,
        default_zone="us-west4-a",
        create_startup_script_filenames=lambda experiment_file_name: (
            "./_install_cuda.sh",
        )
        + create_startup_script_filenames(experiment_file_name),
    ),
}


def pretty_command(command: str) -> str:
    # FYI: written by ChatGPT
    initial_command, *arguments = command.split(" --", 1)
    # Re-add the '--' to the start of each argument
    arguments = "--" + arguments[0] if arguments else ""
    pattern = r"(--\S+?)(?:=(.*?))?(?=\s--|$)"  # matches --argument or --argument=value
    matches = re.findall(pattern, arguments)
    # Re-join command
    delim = " \\\n\t"
    return (
        initial_command
        + delim
        + (delim.join(f"{arg}={val}" if val else arg for arg, val in matches))
    )


def create_instance(
    experiment_file_name: str | None = None,
    zone: str | None = None,
    experiment_type: ExperimentTypes = "gpu",
    project_name: str | None = None,
    instance_name_prefix: str = "instance-pretrain-on-test",
    gcp_service_account_email: str | None = None,
    dont_run_experiment: bool = False,
    is_dry_run: bool = False,
):
    is_experiment_default = experiment_file_name is None
    if not is_experiment_default and dont_run_experiment:
        raise TypeError(
            "Don't provide an experiment file/dir if it's not going to be run."
        )

    # Set up gcloud instance create arguments
    experiment_info = experiment_type_to_info[experiment_type]
    if zone is None:
        zone = experiment_info.default_zone
    if is_experiment_default and dont_run_experiment:
        experiment_file_name = ""  # bleh
        startup_script_filenames = ("./_preamble.sh",)
    else:
        if is_experiment_default:
            experiment_file_name = experiment_info.write_default_experiment_file().name
        else:
            print(f"Creating instance for running {experiment_file_name}")
        startup_script_filenames = experiment_info.create_startup_script_filenames(
            experiment_file_name
        )

    if project_name is None:
        project_name = run_command("gcloud config get-value project").strip("\n")
    instance_name = "-".join(
        (
            instance_name_prefix,
            experiment_type,
            datetime.now().strftime("%Y%m%d%H%M%S"),
            ""
            if is_experiment_default
            else os.path.basename(experiment_file_name)
            .removesuffix(".sh")
            .replace("_", "-"),
        )
    ).rstrip("-")
    if gcp_service_account_email is None:
        gcp_service_account_email = service_account_email()

    # Create instance
    startup_script_file = concatenate_files(startup_script_filenames)
    create_instance_command = experiment_info.create_instance_command(
        project_name=project_name,
        instance_name=instance_name,
        zone=zone,
        gcp_service_account_email=gcp_service_account_email,
        startup_script_filename=startup_script_file.name,
    )
    if is_dry_run:
        print("Here is what the instance will do:")
        sleep(1)  # give some time to process
        print("\n" + run_command(f"cat {startup_script_file.name}") + "\n")
        print("Here is the create instance command:")
        sleep(1)
        print(pretty_command(create_instance_command))
    else:
        create_message = run_command(create_instance_command)
        print("\n" + create_message + "\n")
        post_create_message(project_name, instance_name, zone, dont_run_experiment)


def create_instances(
    experiment_dir: str | None = None,
    zone: str | None = None,
    experiment_type: ExperimentTypes = "gpu",
    project_name: str | None = None,
    instance_name_prefix: str = "instance-pretrain-on-test",
    gcp_service_account_email: str | None = None,
    dont_run_experiment: bool = False,
    is_dry_run: bool = False,
):
    """
    Creates instances which run the experiment files in `experiment_dir`, where one
    instance corresponds to one experiment file.

    Parameters
    ----------
    experiment_dir : str | None, optional
        Directory containing bash .sh files which will be run from the repo root by the
        cloud instance. Assume all cloud and Python env set up is complete. By default,
        ./experiment_mini.sh for "cpu-test / gpu-test" experiment types, else
        ./experiment.sh.
    zone : str | None, optional
        Zone with CPU/GPU availability, by default us-central1-a for CPU and us-west4-a
        for GPU. Consider trying us-west4-b if the default fails.
    experiment_type : ExperimentTypes, optional
        Whether or not this is a CPU test, a full GPU run, or a GPU test (which runs a
        mini experiment with random-weight models). By default, it's a full GPU run.
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
    is_dry_run : bool, optional
        Don't create the instance and run the experiment, just print out what the
        startup script will look like / what the instance will do, and print out the
        create instance command.
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
        create_instance(
            experiment_file_name=experiment_file_name,
            zone=zone,
            experiment_type=experiment_type,
            project_name=project_name,
            instance_name_prefix=instance_name_prefix,
            gcp_service_account_email=gcp_service_account_email,
            dont_run_experiment=dont_run_experiment,
            is_dry_run=is_dry_run,
        )
        print(("-" * os.get_terminal_size().columns) + "\n")


if __name__ == "__main__":
    tapify(create_instances)
