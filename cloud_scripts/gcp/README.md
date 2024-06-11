# Run the experiment in Google Cloud

Note that, in total, the experiment will cost around $40 in compute credits.


## Setup

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
   python launch.py --experiment_type cpu-test
   ```

2. Check that stuff was logged (search for the latest log group with the name `run-`)
   and that data was uploaded to the bucket `pretrain-on-test-accuracies`.

3. Consider deleting these logs:

   ```bash
   python delete_old_test_logs.py
   ```

</details>

## Run the experiment on GPUs

Launch a cloud GPU instance which will run the full experiment, and check that data was
uploaded to GCP. Note that the instance will stop even if there's an error.

1. Run the full experiment (after ensuring your `gcloud` is set to whatever project
   hosts the bucket):

   ```bash
   python launch.py
   ```

   To run multiple experiments in parallel / on multiple instances, put some bash files
   in a directory (e.g.,
   [`./experiments/n500/bert/batch1/`](./experiments/n500/bert/batch1/)) and run:

   ```bash
   python launch.py --experiment_dir experiments/n500/bert/batch1/
   ```

2. Check that stuff was logged (search for the latest log group with the name `run-`)
   and that data was uploaded to the bucket `pretrain-on-test-accuracies`.


## Merge data

After running all of the experiments, merge their data into a single directory which can
be used for analysis. 

1. cd to:

   ```bash
   cd ../../analysis/dirty_file_processing
   ```

2. Copy data from GCP storage to `runs`, for example:

   ```bash
   mkdir -p runs
   ```

   ```bash
   gsutil -m cp -r \
     "gs://pretrain-on-test-accuracies/run-2024-06-03_05-48-48" \
     "gs://pretrain-on-test-accuracies/run-2024-06-03_07-07-16" \
     "gs://pretrain-on-test-accuracies/run-2024-06-04_00-44-44" \
     "gs://pretrain-on-test-accuracies/run-2024-06-04_02-30-36" \
     "gs://pretrain-on-test-accuracies/run-2024-06-04_03-21-06" \
     "gs://pretrain-on-test-accuracies/run-2024-06-04_04-22-47" \
     ./runs
   ```

3. Merge them into a new directory, `accuracies`:

   ```bash
   python merge_runs.py --runs_dir runs --destination_dir accuracies
   ```

4. Verify that the datasets are the same:

   ```bash
   diff <(ls accuracies/n100/bert) <(ls accuracies/n100/gpt2)
   ```

5. When you're ready to analyze this data, copy-paste `accuracies` into the analysis
   dir:

   ```bash
   cp -a accuracies ../
   ```

One day, I might make this process simpler or obsolete.


## Transient errors

These are errors in cloud runs which are not yet handled (b/c I don't know what they
mean or I don't know a reliable way to handle them) but don't seem to impact
correctness.

<details>
<summary>OSError: [Errno 9] Bad file descriptor</summary>

[Link](https://console.cloud.google.com/errors/detail/COfTgoi5qYyDUg?project=virtual-equator-423819-v6).

This error was raised when attempting to upload local CSVs to the bucket. Strange b/c
all of the CSVs were correctly uploaded, and it happened twice in 100 calls across
independent runs.

If you see this, double check that the CSVs for that dataset were uploaded correctly,
and then trigger another experiment run excluding that dataset.

</details>
