"""CLI orchestrator for the data pipeline."""

import argparse
import logging
import sys
import time
from typing import List

from .utils import setup_logging

logger = logging.getLogger("airbnb_pipeline.pipeline")

ALL_STAGES = [
    "ingest",
    "profile",
    "clean",
    "quality",
    "enrich",
    "model",
    "analysis",
    "statistics",
    "business_insights",
    "ml_price_model",
]


def run_pipeline(city_key: str, stages: List[str], force: bool = False) -> None:
    """
    Execute pipeline stages in order.

    Args:
        city_key: City identifier (e.g., 'bangkok').
        stages: List of stage names to execute.
        force: If True, force re-download of data files.
    """
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("Airbnb Market Intelligence Pipeline")
    logger.info("City: %s | Stages: %s | Force: %s", city_key, stages, force)
    logger.info("=" * 60)

    for stage in stages:
        stage_start = time.time()
        logger.info("─" * 40)
        logger.info("STAGE: %s", stage.upper())
        logger.info("─" * 40)

        try:
            if stage == "ingest":
                from .ingest import ingest

                ingest(city_key, force=force)

            elif stage == "profile":
                from .profile import profile

                profile(city_key)

            elif stage == "clean":
                from .clean import clean

                clean(city_key)

            elif stage == "quality":
                from .quality import run_quality_checks

                run_quality_checks(city_key)

            elif stage == "enrich":
                from .enrich import enrich

                enrich(city_key)

            elif stage == "model":
                from .model import build_warehouse

                build_warehouse(city_key)

            elif stage == "analysis":
                from .analysis import run_analysis

                run_analysis(city_key)

            elif stage == "statistics":
                from .statistics import run_statistical_tests

                run_statistical_tests(city_key)

            elif stage == "business_insights":
                from .business_insights import run_business_insights

                run_business_insights(city_key)

            elif stage == "ml_price_model":
                from .ml_price_model import train_and_evaluate_models

                train_and_evaluate_models(city_key)

            else:
                logger.warning("Unknown stage: %s — skipping.", stage)
                continue

            elapsed = time.time() - stage_start
            logger.info("Stage '%s' completed in %.1f seconds.", stage, elapsed)

        except Exception as e:
            elapsed = time.time() - stage_start
            logger.error(
                "Stage '%s' FAILED after %.1f seconds: %s",
                stage,
                elapsed,
                e,
                exc_info=True,
            )
            # Continue to next stage — don't crash the whole pipeline
            logger.warning("Continuing to next stage despite failure.")

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("Pipeline complete. Total time: %.1f seconds.", total_elapsed)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Airbnb Market Intelligence Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available stages: {', '.join(ALL_STAGES)}",
    )
    parser.add_argument(
        "--city",
        required=True,
        help="City key from config/cities.yml (e.g., 'bangkok')",
    )
    parser.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"Comma-separated list of stages to run. Default: all stages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist.",
    )
    parser.add_argument(
        "--include-ml",
        action="store_true",
        help="Include the optional ML price model stage.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    stages = [s.strip().lower() for s in args.stages.split(",")]

    # Remove ML model if flag is not passed
    if not args.include_ml and "ml_price_model" in stages:
        stages.remove("ml_price_model")
    invalid = [s for s in stages if s not in ALL_STAGES]
    if invalid:
        logger.error("Invalid stages: %s. Available: %s", invalid, ALL_STAGES)
        sys.exit(1)

    run_pipeline(args.city, stages, force=args.force)


if __name__ == "__main__":
    main()
