# Run the experiment in Google Cloud

First, [create a bucket](https://cloud.google.com/storage/docs/creating-buckets) called
`pretrain-on-test-accuracies`.

Also consider:

1. Adding an email alert for any errors that pop up during the experiment.

2. [Increasing your quotas](https://console.cloud.google.com/iam-admin/quotas) for
   `GPUS_ALL_REGIONS` and `NVIDIA_T4_GPUS` (default region is `us-west4`) to at least
   1—more than 1 for running experiments in parallel.


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
