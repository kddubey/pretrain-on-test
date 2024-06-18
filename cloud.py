"""
Upload local files and log info to your cloud provider. Currently supports GCP.
"""

import logging
import os
from pathlib import Path
import sys
from typing import Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


###################################### Protocols #######################################


@runtime_checkable
class CreateLogger(Protocol):
    def __call__(self, name: str) -> logging.Logger:
        """
        Returns a `logging.Logger`, maybe w/ a handler which logs to your cloud provider
        under the group/label/tag `name`.
        """


@runtime_checkable
class UploadDirectory(Protocol):
    def __call__(self, directory: str, logger: logging.Logger) -> None:
        """
        Recursively uploads all files in `directory` to your cloud provider.
        """


######################################## Local #########################################


def create_logger_local(name: str) -> logging.Logger:
    """
    Returns a logger which logs to stdout at level INFO.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(filename)-17s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


######################################## GCP ###########################################


def create_logger_gcp(name: str) -> logging.Logger:
    """
    Returns a logger which logs to stdout at level INFO and GCP with
    `python_logger` set to `name`.
    """
    from google.cloud.logging import Client
    from google.cloud.logging.handlers import CloudLoggingHandler

    logger = create_logger_local(name)

    client = Client()
    handler = CloudLoggingHandler(client, name=name)
    logger.addHandler(handler)
    return logger


class UploadGCP:
    """
    Upload to a GCP bucket.
    """

    # Alternate design is to make this one function whose `bucket_name` argument can be
    # partial'd out. Partial objects are opaque, and I'm not sure if re-creating the
    # bucket is bad, so I decided to make it a class.

    def __init__(self, bucket_name: str) -> None:
        from google.cloud.storage import Client

        self.bucket = Client().bucket(bucket_name)

    def upload_directory(
        self, directory: str, logger: logging.Logger, workers: int = 1
    ) -> None:
        """
        Lightly changed from [this snippet](https://github.com/googleapis/python-storage/blob/main/samples/snippets/storage_transfer_manager_upload_directory.py).

        Blob names correspond to the full paths of the files, including the `directory`.
        """
        from google.cloud.storage import transfer_manager

        # First, recursively get all files in `directory` as Path objects.
        directory_as_path_obj = Path(directory)
        paths = directory_as_path_obj.rglob("*")

        # Filter so the list only includes files, not directories themselves.
        file_paths = [path for path in paths if path.is_file()]

        # These paths are relative to the current working directory. Next, make them
        # relative to `directory`
        relative_paths = [path.relative_to(directory) for path in file_paths]

        # Finally, convert them all to strings.
        string_paths = [str(path) for path in relative_paths]

        # Start the upload.
        blob_name_prefix = directory if directory.endswith("/") else directory + "/"
        results = transfer_manager.upload_many_from_filenames(
            self.bucket,
            string_paths,
            source_directory=directory,
            blob_name_prefix=blob_name_prefix,
            max_workers=workers,
        )
        for name, result in zip(string_paths, results, strict=True):
            blob_name = blob_name_prefix + name
            # The results list is either `None` or an exception for each filename in
            # the input list, in order.
            if isinstance(result, Exception):
                if isinstance(result, OSError) and result.errno == 9:
                    logger.warning(
                        f"'Bad file descriptor' when attempting to upload {blob_name} "
                        f"to GCP bucket {self.bucket.name}. Please double check that "
                        "it was uploaded correctly."
                    )
                else:
                    logger.error(
                        f"Failed to upload {blob_name} to GCP bucket {self.bucket.name}"
                    )
                    raise result
            else:
                logger.info(f"Uploaded {blob_name} to GCP bucket {self.bucket.name}")


######################################## Export ########################################


class DataHandlers(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    # Pydantic stuff: extra attributes are not allowed, and the object is immutable

    create_logger: CreateLogger
    upload_directory: UploadDirectory


CloudProviders = Literal[None, "gcp"]


do_nothing = lambda *args, **kwargs: None


cloud_provider_to_create_data_handlers: dict[
    CloudProviders, Callable[..., DataHandlers]
] = {
    None: lambda: DataHandlers(
        create_logger=create_logger_local,
        upload_directory=do_nothing,
    ),
    # They're lambdas so that evaluation is delayed; cloud-specific modules aren't
    # imported and cloud-specific env vars aren't checked until needed
    "gcp": lambda bucket_name=None: DataHandlers(
        create_logger=create_logger_gcp,
        upload_directory=UploadGCP(
            bucket_name=bucket_name or os.environ["PRETRAIN_ON_TEST_BUCKET_NAME"]
        ).upload_directory,
    ),
}
