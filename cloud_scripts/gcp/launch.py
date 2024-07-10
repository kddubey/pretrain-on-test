import atexit
from contextlib import redirect_stderr
from datetime import datetime
from functools import partial, wraps
import io
from itertools import cycle
import os
import re
import shlex
import subprocess
import sys
import tempfile
from time import sleep
from typing import (
    Callable,
    Generic,
    Iterator,
    Literal,
    NoReturn,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict
import rich
from tap import tapify

import _zones


######################################## Utils #########################################


def run_command(command: str) -> str:
    result = subprocess.run(shlex.split(command), capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
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


def _write_temp_file_default(filename: str):
    print(f"Creating instance for running {filename}")
    lines = (f"{filename}\n",)
    return write_temp_file(lines)


def write_experiment_mini():
    return _write_temp_file_default("./experiment_mini.sh")


def write_experiment_full():
    return _write_temp_file_default("./experiment.sh")


def sh_file_no_default():
    raise ValueError(
        "There is no default sh file for this run type. Please supply the "
        "sh_dir_or_filename argument."
    )


################################# GCP-specific things ##################################


# TODO: The GCP Python API is not very nice. But I should use that instead


def create_instance_command_cpu(
    instance_name: str,
    project_name: str,
    zone: str,
    gcp_service_account_email: str,
    startup_script_filename: str,
    machine_type: str = "e2-standard-2",
    boot_disk_size: int = 80,
):
    return (
        f"gcloud compute instances create {instance_name} "
        f"--project={project_name} "
        f"--zone={zone} "
        f"--machine-type={machine_type} "
        f"--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default "
        f"--service-account={gcp_service_account_email} "
        f"--scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write "
        f"--create-disk=auto-delete=yes,boot=yes,device-name={instance_name},image=projects/debian-cloud/global/images/debian-12-bookworm-v20240515,mode=rw,size={boot_disk_size},type=projects/{project_name}/zones/{zone}/diskTypes/pd-balanced "
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
    gpu_type: Literal["T4", "L4", "A100"] = "T4",
    boot_disk_size: int = 80,
):
    gpu_type_to_machine_type = {
        "T4": "n1-highmem-2",
        "L4": "g2-standard-4",
        "V100": "n1-highmem-2",
        "A100": "a2-highgpu-1g",
    }
    gpu_type_to_accelerator_type = {
        "T4": "nvidia-tesla-t4",
        "L4": "nvidia-l4",
        "V100": "nvidia-tesla-v100",
        "A100": "nvidia-tesla-a100",
    }
    return (
        f"gcloud compute instances create {instance_name} "
        f"--project={project_name} "
        f"--zone={zone} "
        f"--machine-type={gpu_type_to_machine_type[gpu_type]} "
        f"--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default "
        f"--maintenance-policy=TERMINATE "
        f"--provisioning-model=STANDARD "
        f"--service-account={gcp_service_account_email} "
        f"--scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write "
        f"--accelerator=count=1,type={gpu_type_to_accelerator_type[gpu_type]} "
        f"--create-disk=auto-delete=yes,boot=yes,device-name={instance_name},image=projects/ml-images/global/images/c2-deeplearning-pytorch-2-2-cu121-v20240514-debian-11,mode=rw,size={boot_disk_size},type=projects/{project_name}/zones/{zone}/diskTypes/pd-balanced "
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
    something_went_wrong_exception = ValueError(
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
    except subprocess.CalledProcessError as command_exception:
        # missing perms or wrong display name
        raise something_went_wrong_exception from command_exception
    else:
        if email and "@" in email:
            return email
        else:
            raise something_went_wrong_exception


def post_create_message(
    project_name: str, instance_name: str, zone: str, just_create: bool
) -> None:
    print(
        "View raw logs here (cleaner logs are in the Logs Explorer page):\n"
        f"https://console.cloud.google.com/compute/instancesDetail/zones/{zone}"
        f"/instances/{instance_name}?project={project_name}&tab=monitoring&pageState="
        f"(%22timeRange%22:(%22duration%22:%22PT1H%22),%22observabilityTab%22:(%22main"
        "Content%22:%22logs%22))"
    )
    print()
    if just_create:
        print(
            "To SSH into the instance, wait 1-2 minutes and then (in a new terminal "
            "tab) run:\n"
            f'gcloud compute ssh --zone "{zone}" "{instance_name}" --project '
            f'"{project_name}"'
        )
        print()


######################################### Main #########################################


def create_startup_script_filenames(sh_file_name: str):
    return (
        "./_preamble.sh",
        "../_setup_python_env.sh",
        sh_file_name,
        "../_shutdown.sh",
    )


def create_startup_script_filenames_gpu(sh_file_name: str):
    return ("./_install_cuda.sh",) + create_startup_script_filenames(sh_file_name)


def create_startup_script_filenames_analysis(sh_file_name: str):
    return (
        "./_preamble.sh",
        "../_setup_python_env.sh",
        "../_setup_python_env_analysis.sh",
        sh_file_name,
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


class InstanceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    # Pydantic stuff: extra attributes are not allowed, and the object is immutable

    # The full instance name will add the run type, a timestamp, and the name of the
    # file that's being run, if applicable
    create_instance_command: CreateInstanceCommand
    default_zone: str
    create_startup_script_filenames: Callable[[str], tuple[str]]
    instance_name_prefix: str = "instance-pretrain-on-test"
    write_default_sh_file: Callable[[], tempfile._TemporaryFileWrapper | NoReturn] = (
        sh_file_no_default
    )


RunTypes = Literal[
    "cpu-test",
    "gpu-test",
    "gpu-t4",
    "gpu-l4",
    "gpu-v100",
    "gpu-a100",
    "analysis",
]


run_type_to_info: dict[RunTypes, InstanceInfo] = {
    "cpu-test": InstanceInfo(
        create_instance_command=create_instance_command_cpu,
        default_zone="us-central1-a",
        create_startup_script_filenames=create_startup_script_filenames,
        write_default_sh_file=write_experiment_mini,
    ),
    "gpu-test": InstanceInfo(
        create_instance_command=partial(create_instance_command_gpu, gpu_type="T4"),
        default_zone="us-west4-a",
        create_startup_script_filenames=create_startup_script_filenames_gpu,
        write_default_sh_file=write_experiment_mini,
    ),
    "gpu-t4": InstanceInfo(
        create_instance_command=partial(create_instance_command_gpu, gpu_type="T4"),
        default_zone="us-west4-a",
        create_startup_script_filenames=create_startup_script_filenames_gpu,
        write_default_sh_file=write_experiment_full,
    ),
    "gpu-l4": InstanceInfo(
        create_instance_command=partial(create_instance_command_gpu, gpu_type="L4"),
        default_zone="us-west4-a",
        create_startup_script_filenames=create_startup_script_filenames_gpu,
        write_default_sh_file=write_experiment_full,
    ),
    "gpu-v100": InstanceInfo(
        create_instance_command=partial(create_instance_command_gpu, gpu_type="V100"),
        default_zone="us-west4-a",
        create_startup_script_filenames=create_startup_script_filenames_gpu,
        write_default_sh_file=write_experiment_full,
    ),
    "gpu-a100": InstanceInfo(
        create_instance_command=partial(create_instance_command_gpu, gpu_type="A100"),
        default_zone="us-west4-a",
        create_startup_script_filenames=create_startup_script_filenames_gpu,
        write_default_sh_file=write_experiment_full,
    ),
    "analysis": InstanceInfo(
        create_instance_command=partial(
            create_instance_command_cpu,
            machine_type="e2-highmem-8",
            boot_disk_size=100,
        ),
        default_zone="us-central1-a",
        create_startup_script_filenames=create_startup_script_filenames_analysis,
        write_default_sh_file=sh_file_no_default,
    ),
}


def pretty_command(command: str) -> str:
    # FYI: written mostly by ChatGPT. Seems to be correct after some offline tests
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
    *,
    sh_file_name: str | None = None,
    zone: str | None = None,
    run_type: RunTypes = "gpu-t4",
    project_name: str | None = None,
    gcp_service_account_email: str | None = None,
    just_create: bool = False,
    dry_run: bool = False,
) -> None:
    is_sh_file_default = sh_file_name is None
    if not is_sh_file_default and just_create:
        raise TypeError("Don't provide an sh file/dir if it's not going to be run.")

    # Set up gcloud instance create arguments
    instance_info = run_type_to_info[run_type]
    if zone is None:
        zone = instance_info.default_zone
    if is_sh_file_default and just_create:
        sh_file_name = ""  # bleh
        startup_script_filenames = ("./_preamble.sh",)
    else:
        if is_sh_file_default:
            sh_file_name = instance_info.write_default_sh_file().name
        else:
            print(f"Creating instance for running {sh_file_name}")
        startup_script_filenames = instance_info.create_startup_script_filenames(
            sh_file_name
        )

    if project_name is None:
        project_name = run_command("gcloud config get-value project").strip("\n")
    instance_name = "-".join(
        (
            instance_info.instance_name_prefix,
            run_type,
            datetime.now().strftime("%Y%m%d%H%M%S"),
            ""
            if is_sh_file_default
            else os.path.basename(sh_file_name).removesuffix(".sh").replace("_", "-"),
        )
    ).rstrip("-")
    if gcp_service_account_email is None:
        gcp_service_account_email = service_account_email()

    # Create instance
    startup_script_file = concatenate_files(startup_script_filenames)
    create_instance_command = instance_info.create_instance_command(
        project_name=project_name,
        instance_name=instance_name,
        zone=zone,
        gcp_service_account_email=gcp_service_account_email,
        startup_script_filename=startup_script_file.name,
    )
    if dry_run:
        print("Here is what the instance will do:")
        sleep(1)  # give some time to process
        print("\n" + run_command(f"cat {startup_script_file.name}") + "\n")
        print("Here is the create instance command:")
        sleep(1)
        print(pretty_command(create_instance_command))
    else:
        create_message = run_command(create_instance_command)
        print("\n" + create_message + "\n")
        post_create_message(project_name, instance_name, zone, just_create)


_T = TypeVar("_T")


class _Cycler(Generic[_T]):
    def __init__(self, cycle: "cycle[_T]") -> None:
        self._cycle = cycle

    @property
    def current(self):
        if not hasattr(self, "_current"):
            return next(self)
        return self._current

    def __iter__(self) -> Iterator[_T]:
        return self

    def __next__(self):
        self._current = next(self._cycle)
        return self._current


_gpu_type_to_zone_cycle = {
    "T4": cycle(_zones.t4_gpu_zones),
    "L4": cycle(_zones.l4_gpu_zones),
}


gpu_type_to_zone_cycler = {
    zone: _Cycler(zone_cycle) for zone, zone_cycle in _gpu_type_to_zone_cycle.items()
}


def try_zones(create_instance: Callable, gpu_type: Literal["T4", "L4"] = "T4"):
    """
    Decorator which tries to create an instance anywhere in the world with `gpu_type`
    availability.
    """
    no_resources_available_codes = (
        "code: ZONE_RESOURCE_POOL_EXHAUSTED",
        "code: ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS",
    )  # ty https://github.com/doitintl/gpu-finder/issues/1

    @wraps(create_instance)
    def wrapper(**kwargs):
        # To iterate through zones, we're not going to do:
        #
        # for zone in cycle(t4_gpu_zones)
        #
        # i.e., we're not going to start the cycle from whatever the start of
        # t4_gpu_zones is. We should instead start from whatever zone last worked, b/c
        # if this zone had availability for 1 instance, it probably has availability for
        # more. We'll stay in that zone for future instances we need to create until it
        # runs out, at which point we'll move to the next zone in the cycle. This means
        # we need state across function calls. This state is global so that any call
        # from anywhere will try the most recently available zone.
        zone = gpu_type_to_zone_cycler[gpu_type].current
        while True:
            print(f"Trying {zone}...")
            kwargs["zone"] = zone
            f = io.StringIO()
            try:
                with redirect_stderr(f):
                    return create_instance(**kwargs)
            except subprocess.CalledProcessError as exception:
                stderr = f.getvalue()
                print(stderr)
                if any(code in stderr for code in no_resources_available_codes):
                    print(
                        "Clearing your terminal screen and auto-retrying the next zone "
                        "in 5 seconds..."
                    )
                    sleep(5)
                    os.system("cls" if os.name == "nt" else "clear")
                    zone = next(gpu_type_to_zone_cycler[gpu_type])
                else:
                    raise exception

    return wrapper


def all_sh_files(sh_dir_or_filename: str) -> list[str]:
    """
    Also checks that there are `*.sh` files here. If there aren't raises a `ValueError`.
    """
    if os.path.isdir(sh_dir_or_filename):
        file_names = [
            os.path.join(sh_dir_or_filename, filename)
            for filename in sorted(os.listdir(sh_dir_or_filename))
        ]
    else:
        file_names = [sh_dir_or_filename]
    return [file_name for file_name in file_names if file_name.endswith(".sh")]


def move_sh_file_to_in_progress_dir(
    sh_file_name: str, sh_dir: str, dry_run: bool = False
) -> None:
    in_progress_dir = os.path.join(sh_dir, "in_progress")
    if not os.path.exists(in_progress_dir) and not dry_run:
        os.mkdir(in_progress_dir)
    in_progress_file = os.path.join(in_progress_dir, os.path.basename(sh_file_name))
    if not dry_run:
        os.rename(sh_file_name, in_progress_file)
    print(f"\nMoved {sh_file_name} to {in_progress_file}\n")


def create_instances(
    sh_dir_or_filename: str | None = None,
    zone: str | None = None,
    run_type: RunTypes = "gpu-t4",
    project_name: str | None = None,
    gcp_service_account_email: str | None = None,
    just_create: bool = False,
    dry_run: bool = False,
    any_zone: bool = False,
):
    """
    Creates instances which run the sh file(s) in `sh_dir_or_filename`, where one
    instance corresponds to one sh file.

    Parameters
    ----------
    sh_dir_or_filename : str | None, optional
        Either a *.sh file, or a directory including *.sh files which will be run by the
        cloud instance. Assume all cloud and Python env set up is complete. By default,
        the sh file is ./experiment_mini.sh for "cpu-test / gpu-test" run types, else
        ./experiment.sh (at the repo root).
    zone : str | None, optional
        Zone with CPU/GPU availability, by default us-central1-a for CPU and us-west4-a
        for GPU. Consider trying us-west4-b if the default fails.
    run_type : RunTypes, optional
        Whether or not this is a CPU test, a GPU test (which runs a mini experiment with
        random-weight models), a full GPU run, or an analysis. By default, it's a full
        GPU run.
    project_name : str | None, optional
        Name of the project for this run, by default `gcloud config get-value project`.
    gcp_service_account_email : str | None, optional
        Your Google Cloud service account. By default it's the one with display name
        "Compute Engine default service account".
    just_create : bool, optional
        Just set up the instance for SSHing into instead of running any sh files. By
        default, the instance will run the sh files in sh_dir_or_filename.
    dry_run : bool, optional
        Don't create the instance and run the experiment, just print out what the
        startup script will look like / what the instance will do, and print out the
        create instance command.
    any_zone : bool, optional
        Try to find any zone in the world with the relevant resource (T4 GPU for GPU
        runs) available. **Adding this flag will cause the script to run indefinitely.**
        It'll keep running until it creates an instance for each sh file in
        `sh_dir_or_filename`. Go do something else.
    """
    # Extract all sh files from sh_dir_or_filename
    if sh_dir_or_filename is None:
        sh_file_names = (None,)
    else:
        sh_file_names = all_sh_files(sh_dir_or_filename)
        if not sh_file_names:
            raise ValueError(f"Expected *.sh files in {sh_dir_or_filename}.")
        print("Attempting to create instances for the following files:")
        print("\n".join(sh_file_names))
        print(("-" * os.get_terminal_size().columns) + "\n")

    gpu_type = "T4"
    create_instance_func = (
        try_zones(create_instance, gpu_type) if any_zone else create_instance
    )

    try:
        for sh_file_name in sh_file_names:
            create_instance_func(
                sh_file_name=sh_file_name,
                zone=zone,
                run_type=run_type,
                project_name=project_name,
                gcp_service_account_email=gcp_service_account_email,
                just_create=just_create,
                dry_run=dry_run,
            )
            if os.path.isdir(sh_dir_or_filename):
                # To "auto-batch" runs, use the fact that the create_instance_func will
                # raise an exception when the quota is exceeded.
                move_sh_file_to_in_progress_dir(
                    sh_file_name, sh_dir_or_filename, dry_run=dry_run
                )
            print(("-" * os.get_terminal_size().columns) + "\n")
    finally:
        if sh_dir_or_filename is None or not os.path.isdir(sh_dir_or_filename):
            return
        # Successfully ran files were moved to sh_dir/in_progress/. So check if there
        # are more in sh_dir
        if remaining_sh_files := all_sh_files(sh_dir_or_filename):
            rich.print("[bold red]WARNING: Not every sh file was run.[/bold red]")
            print(
                "To run them (after you have instance capacity), run the same launch "
                "command you just ran."
            )
            print("Remaining sh files to run:")
            print("\n".join(remaining_sh_files))
            print()


if __name__ == "__main__":
    tapify(create_instances)
