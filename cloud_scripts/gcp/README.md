# Run the experiment in Google Cloud

Note that, in total, the experiment will cost around $200 in compute credits.


## Setup

First, [create a bucket](https://cloud.google.com/storage/docs/creating-buckets) called
`pretrain-on-test-accuracies`.

Also consider:

1. Adding an [alert](https://cloud.google.com/monitoring/support/notification-options)
   to notify you if any errors pop up during a cloud run.

2. [Increasing your quotas](https://console.cloud.google.com/iam-admin/quotas) for
   `GPUS_ALL_REGIONS` and `NVIDIA_T4_GPUS` to 1 (or `q > 1` for running experiments in
   parallel) and `SSD_TOTAL_GB` to `250 * q`. The default region is `us-west4`. Or make
   quota requests according to error messages. FYI I couldn't get my quota past 4 GPUs.

3. [Adding a
   secret](https://cloud.google.com/secret-manager/docs/creating-and-accessing-secrets#secretmanager-create-secret-console),
   `HF_TOKEN`, which has your Hugging Face login token if you need to use Mistral or
   other models which require this authorization before downloading weights. Then [give
   your service account permission to access this
   secret](https://cloud.google.com/secret-manager/docs/access-control).


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
uploaded to GCP.

1. Run the mini CPU test (after ensuring your `gcloud` is set to whatever project hosts
   the bucket):

   ```bash
   python launch.py --run_type cpu-test
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
   [`./experiments/m100/n500/bert/`](./experiments/m100/n500/bert/)) and run:

   ```bash
   python launch.py --sh_dir_or_filename experiments/m100/n500/bert/
   ```

   If you're getting an error with code `ZONE_RESOURCE_POOL_EXHAUSTED` (b/c there aren't
   any T4 GPUs available in the requested zone), then consider adding the flag
   `--any_zone`, which will automatically try to find a zone with availability.

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
   cd runs
   ```

   ```bash
   gsutil -m cp -r \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_06-52-44-m50_n100_gpt2_4" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_06-52-52-m50_n100_gpt2_2" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_06-52-52-m50_n100_gpt2_5" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_06-53-09-m50_n100_gpt2_7" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_14-23-58-m50_n100_gpt2_6" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_14-24-05-m50_n100_gpt2_3" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_14-24-48-m50_n100_gpt2_1" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_19-59-41-m50_n100_bert_2" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_20-18-05-m50_n100_bert_4" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_20-19-25-m50_n100_bert_6" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_20-21-02-m50_n100_bert_5" \
     "gs://pretrain-on-test-accuracies/run-2024-06-17_21-46-23-m50_n100_bert_7" \
     "gs://pretrain-on-test-accuracies/run-2024-06-18_00-25-45-m50_n100_bert_1" \
     "gs://pretrain-on-test-accuracies/run-2024-06-18_03-00-59-m50_n100_bert_3" \
     .
   ```

   ```bash
   cd ..
   ```

3. Merge them into a new directory, `accuracies`:

   ```bash
   python merge_runs.py --runs_dir runs --destination_dir accuracies
   ```

4. Verify that the datasets are the same:

   ```bash
   diff <(ls accuracies/m50/n100/bert) <(ls accuracies/m50/n100/gpt2)
   ```

5. When you're ready to analyze this data, copy-paste (or move, whatever you prefer)
   `accuracies` into the analysis dir:

   ```bash
   cp -a accuracies ../
   ```


## Run the analysis

All of the analyses can be run locally, but I was hitting performance issues for $n =
50$ and $100$ b/c the number of subsamples for each dataset is $100$. Also, multicore
isn't working locally. I ran it in the cloud instead.

Launch a high-memory, 4-core CPU instance which will run the analyses, e.g., in
[`./analyses/m100`](./analyses/m100):

```bash
python launch.py \
   --run_type analysis \
   --sh_dir_or_filename analyses/m100
```
