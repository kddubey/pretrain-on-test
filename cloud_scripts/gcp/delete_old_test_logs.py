import os
import subprocess

from tap import tapify
from tqdm.auto import tqdm

from launch import run_command


def main():
    """
    Delete all log entries from log groups with the name run-*cpu-test*
    """
    # Seems like the Python API requires listing every log entry:
    # https://stackoverflow.com/questions/45760906/retrieve-list-of-log-names-from-google-cloud-stackdriver-api-with-python
    # Gonna use CLI instead
    list_log_names_result = run_command("gcloud logging logs list")
    log_names_all = (
        list_log_names_result.removeprefix("NAME\n").removesuffix("\n").split("\n")
    )

    def is_cpu_test_run(log_name_full: str) -> bool:
        log_name = log_name_full.split("/")[-1]
        return log_name.startswith("run-") and "cpu-test" in log_name

    log_names_delete = [
        log_name_full
        for log_name_full in log_names_all
        if is_cpu_test_run(log_name_full)
    ]
    if not log_names_delete:
        print("There are no test logs to delete.")
        exit()

    print(
        "All entries from the following logs will be deleted:\n"
        + "\n".join(log_names_delete)
    )
    message_prompt = "\nEnter y to delete them, n to cancel: [y/n] "
    does_user_approve = None
    while does_user_approve not in ["y", "n"]:
        does_user_approve = input(message_prompt)
    if does_user_approve == "n":
        exit()

    command = ["gcloud", "logging", "logs", "delete"]

    for log_name in tqdm(log_names_delete):
        print()
        print("-" * os.get_terminal_size().columns)
        delete_log = command + [log_name]
        process = subprocess.Popen(
            delete_log,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input="Y\n")
        if stdout:
            print("Output:", stdout)
        if stderr:
            print("Errors:", stderr)
        print("-" * os.get_terminal_size().columns)
        print()


if __name__ == "__main__":
    tapify(main)
