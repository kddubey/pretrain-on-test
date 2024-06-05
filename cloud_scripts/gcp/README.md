# Run the experiment in Google Cloud

First, [create a bucket](https://cloud.google.com/storage/docs/creating-buckets) called
`pretrain-on-test-accuracies`.

Also consider:

1. Adding an email alert for any errors that pop up during the experiment.

2. [Increasing your quotas](https://console.cloud.google.com/iam-admin/quotas) for
   `GPUS_ALL_REGIONS` and `NVIDIA_T4_GPUS` to at least 1 (or `n > 1` for running
   experiments in parallel) and `SSD_TOTAL_GB` to `250 * n`. The default region is
   `us-west4`.


<details>
<summary>Consider locally testing that cloud logging and storage works</summary>

Run a mini experiment on your computer and check that data was uploaded to GCP.

1. Install the `gcp` requirements (at the repo root):

   ```bash
   python -m pip install ".[gcp]"
   ```

2. From the repo root, run the mini CPU test (after ensuring your `gcloud` is set to
   whatever project hosts the bucket):

   ```bash
   PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp" \
   PRETRAIN_ON_TEST_BUCKET_NAME="pretrain-on-test-accuracies" \
   ./experiment_mini.sh
   ```

3. Check that stuff was logged (search for the latest log group with the name `run-`)
   and that data was uploaded to the bucket `pretrain-on-test-accuracies`.

</details>

<details>
<summary>Test that cloud launches work</summary>

Launch a cloud instance which will run a mini experiment, and check that data was
uploaded to GCP. Note that the instance will stop even if there's an error.

1. Run the mini CPU test (after ensuring your `gcloud` is set to whatever project hosts
   the bucket):

   ```bash
   python launch.py --is_cpu_test
   ```

2. Check that stuff was logged (search for the latest log group with the name `run-`)
   and that data was uploaded to the bucket `pretrain-on-test-accuracies`.

3. Consider deleting these logs:

   ```bash
   python delete_old_test_logs.py
   ```

</details>

<details>
<summary>Run the full experiment on GPU</summary>

Launch a cloud GPU ($$) instance which will run the full experiment, and check that data
was uploaded to GCP. Note that the instance will stop even if there's an error.

1. Run the GPU script (after ensuring your `gcloud` is set to whatever project hosts the
   bucket):

   ```bash
   python launch.py
   ```

2. Check that stuff was logged (search for the latest log group with the name `run-`)
   and that data was uploaded to the bucket `pretrain-on-test-accuracies`.

</details>


## Transient errors

These are errors which are not yet handled (b/c I don't know what they mean or I don't
know a reliable way to handle them) but don't seem to impact correctness.

<details>
<summary>OSError: [Errno 9] Bad file descriptor</summary>

[Link](https://console.cloud.google.com/errors/detail/COfTgoi5qYyDUg?project=virtual-equator-423819-v6).

This error was raised when attempting to upload local CSVs to the bucket. Strange b/c
all of the CSVs were correctly uploaded, and it happened once in 50 calls across
independent runs.

If you see this, double check that the CSVs for that dataset were uploaded correctly,
and then trigger another experiment run excluding that dataset.

</details>
