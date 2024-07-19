"""
Fit the hierarchical model, and save its inference data.
"""

from datetime import datetime
import os
import sys

from pydantic import BaseModel, ConfigDict, Field
from tap import tapify

import utils

# sys hack to import from parent
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import cloud
from cloud import do_nothing


class Analysis(BaseModel):
    """
    Analysis configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    # Pydantic stuff: extra attributes are not allowed, and the object is immutable

    num_train: int = Field(
        description=(
            "Indicator for the dataset of accuracies you wanna analyze, e.g., m = 50 or"
            "100"
        )
    )
    num_test: int = Field(
        description=(
            "Indicator for the dataset of accuracies you wanna analyze, e.g., n = 50, "
            "100, 200, or 500"
        )
    )
    equation: str = Field(
        default="p(num_correct, num_test) ~ method + lm_type + (1|dataset/method) + (1|dataset/pair)",
        description="Model equation",
    )
    id_vars: tuple[str, ...] = Field(
        default=("num_test", "pair", "lm_type", "dataset"),
        description="Ya know, the id_vars, duh",
    )
    analysis_name: str = Field(
        default="",
        description=(
            "Name of the analysis, in case it helps you remember what changed. If "
            "supplied, this name gets appended to the analysis ID string: "
            "analysis-{timestamp}-{run_name}"
        ),
    )
    accuracies_home_dir: str = Field(
        default="accuracies_from_paper",
        description="Where the accuracy data is stored, relative to the analysis dir",
    )
    cores: int = Field(
        default=4,
        description="The number of CPU cores for running the analysis in parallel",
    )


def run(
    analysis: Analysis,
    create_logger: cloud.CreateLogger = cloud.create_logger_local,
    upload_directory: cloud.UploadDirectory = do_nothing,
) -> None:
    """
    Run the experiment.

    Parameters
    ----------
    analysis : Analysis
        configuration for the analysis
    create_logger : cloud.CreateLogger, optional
        Callable which takes as input a single argument for the name of the log
        group/label/tag, and outputs a `logging.Logger` object. By default, a logger is
        created which only logs to stdout.
    upload_directory : cloud.UploadDirectory, optional
        Callable which takes as input `directory` and `logger` arguments and uploads all
        local content in `directory` somewhere else, e.g., S3. By default, nothing is
        uploaded.
    """
    # Meta info
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    analysis_id = (
        f"analysis-{current_time}"
        f"{'-' + analysis.analysis_name if analysis.analysis_name else ''}"
    )

    # Create logger
    logger = create_logger(analysis_id)
    logger.info(f"ID of the run: {analysis_id}")
    logger.info(analysis)

    try:
        # Load data
        num_correct_df = utils.load_all_num_correct(
            os.path.join(analysis.accuracies_home_dir, f"m{analysis.num_train}"),
            analysis.num_test,
        )

        # logger.info("Analyzing extra - base")
        # _, summary_control, _ = utils.stat_model(
        #     num_correct_df,
        #     treatment="extra",
        #     control="base",
        #     equation=analysis.equation,
        #     id_vars=analysis.id_vars,
        #     cores=analysis.cores,
        #     plot=False,
        # )
        # logger.info("Writing control inference data")
        # os.mkdir(analysis_id)
        # summary_control.to_netcdf(
        #     filename=os.path.join(
        #         analysis_id,
        #         f"main_m{analysis.num_train}_n{analysis.num_test}_control.nc",
        #     )
        # )
        # upload_directory(analysis_id, logger)
        # # Free up some memory? idk
        # del _
        # del summary_control

        # print("\n" + ("#" * 50) + "\n")

        logger.info("Analyzing test - extra")
        _, summary_bias, _ = utils.stat_model(
            num_correct_df,
            treatment="test",
            control="extra",
            equation=analysis.equation,
            id_vars=analysis.id_vars,
            cores=analysis.cores,
            plot=False,
        )
        logger.info("Writing treatment inference data")
        os.mkdir(analysis_id)
        summary_bias.to_netcdf(
            filename=os.path.join(
                analysis_id,
                f"main_m{analysis.num_train}_n{analysis.num_test}_treatment.nc",
            )
        )
        upload_directory(analysis_id, logger)
    except Exception as exception:
        logger.error(exception, exc_info=True)
        raise


if __name__ == "__main__":
    analysis = tapify(Analysis)
    cloud_provider = os.environ.get("PRETRAIN_ON_TEST_CLOUD_PROVIDER")
    create_data_handlers = cloud.cloud_provider_to_create_data_handlers[cloud_provider]
    data_handlers = create_data_handlers()
    run(
        analysis,
        create_logger=data_handlers.create_logger,
        upload_directory=data_handlers.upload_directory,
    )
