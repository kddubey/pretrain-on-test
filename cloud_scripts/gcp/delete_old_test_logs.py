"""
Dumb script which deletes old test logs.
"""

import subprocess
from tqdm.auto import tqdm


# Seems like the Python API requires listing every entry:
# https://stackoverflow.com/questions/45760906/retrieve-list-of-log-names-from-google-cloud-stackdriver-api-with-python
list_log_names_command = ["gcloud", "logging", "logs", "list"]
list_log_names_result = subprocess.run(
    list_log_names_command, capture_output=True, text=True
)
log_names_all = (
    list_log_names_result.stdout.removeprefix("NAME\n").removesuffix("\n").split("\n")
)


def is_cpu_test_run(log_name_full: str) -> bool:
    # match "run-*-cpu-test-*" w/o regex
    log_name = log_name_full.split("/")[-1]
    log_name_split = log_name.split("-")
    return log_name_split[0] == "run" and log_name_split[-3:-1] == ["cpu", "test"]


log_names_delete = [
    log_name_full for log_name_full in log_names_all if is_cpu_test_run(log_name_full)
]
if not log_names_delete:
    print("There are no test logs to delete.")
    exit()


print("The following logs will be deleted:\n" + "\n".join(log_names_delete))
message_prompt = "\nEnter y to delete them, n to cancel [y/n]: "
does_user_approve = None
while does_user_approve not in ["y", "n"]:
    does_user_approve = input(message_prompt)
if does_user_approve == "n":
    exit()


command = ["gcloud", "logging", "logs", "delete"]

for log_name in tqdm(log_names_delete):
    print()
    print("-" * 20)
    delete_log = command + [log_name]
    process = subprocess.Popen(
        delete_log,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(input="Y\n")
    print("Output:", stdout)
    print("Errors:", stderr)
    print("-" * 20)
    print()
