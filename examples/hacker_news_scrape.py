import time
from dataclasses import dataclass
from typing import List
import os
from pathlib import Path

from conduit.compute_provider.local import LocalNetworkBinding
from conduit.runtime import LMLiteBlock
from conduit import LmLiteModelConfig, ComputeProvider
from conduit.blocks import FileSystemWriteBlock, HttpGetBlock, Sqlite3Block
from conduit.utils.deployment import DeploymentConstraint
from conduit.compute_provider.runpod.runpod_types import GPUS


# # -----------------------------------------------------------------------------
# Demo: Cloud GPU deployment (Runpod) + end-to-end “AI glue” pipeline with Conduit
#
# This example showcases running an LMLite model on Runpod GPU infrastructure
# (instead of LOCAL), and using the model as a “parser + code generator” to move
# data between real systems:
#
#   HTTP -> LLM structured extraction -> LLM SQL generation -> SQLite write
#
# What it does:
#   1) Deploys a model via LMLiteBlock on ComputeProvider.RUNPOD, using placement
#      constraints (e.g., ENTERPRISE + SINGLE_DEVICE for compliant capacity).
#   2) Fetches Hacker News HTML via HttpGetBlock.
#   3) Uses the model to extract the front-page articles into a typed list
#      (title / points / link), producing structured data from raw HTML.
#   4) If a schema file doesn’t exist, asks the model to generate a SQLite schema
#      for an `articles` table and writes it to disk (schema.sql).
#   5) Uses the model again to transform the structured article list into a SQL
#      INSERT statement, then executes it with Sqlite3Block into hackernews.db.
#
# Notes / gotchas:
#   - This intentionally demonstrates cloud placement controls; unlike LOCAL,
#     Runpod deployments can be filtered via DeploymentConstraint and may be
#     hardware-selected by the provider/scheduler.
#   - The SQL generation in this snippet is NOT parameterized / escaped; it’s a
#     demo and can break (or be unsafe) if inputs contain quotes or weird chars.
#     Prefer parameterized inserts or strict validation in real usage.
# -----------------------------------------------------------------------------


def test_hacker_news_scrape(model: str = "Qwen/Qwen3-4B-Instruct-2507") -> None:
    """
    Scrapes the Hacker News front page and stores parsed articles into a SQLite DB using LMLite.
    """

    # Callable that returns an HTTP operation
    hacker_news_get_block = HttpGetBlock(endpoint="https://news.ycombinator.com/")

    # LMLite deployment block: defines what to run, where to run it, and how to place/scale it.
    lm_lite_block = LMLiteBlock(
        models=[
            # List as many models as you want. Conduit/LMLite will validate feasibility
            # (VRAM/compute) and error if the request can’t be satisfied.
            LmLiteModelConfig(
                "Qwen/Qwen3-4B-Instruct-2507",  # Hugging Face model id
                max_model_len=262144,  # Configure runtime for this max context length
                max_model_concurrency=1,  # Per-replica concurrency / request pool size
            ),
        ],
        # Compute provider (where the deployment runs)
        compute_provider=ComputeProvider.RUNPOD,
        # --- Runpod cloud selection quirk ---
        # Runpod has multiple cloud pools (e.g., ENTERPRISE/secure vs COMMUNITY).
        #
        # - constraints=[DeploymentConstraint.ENTERPRISE] filters placement to enterprise-eligible capacity.
        # - Switching to COMMUNITY on Runpod requires a provider override:
        #     compute_provider_config_overrides={"cloudType": "COMMUNITY"}
        #   And you must REMOVE the ENTERPRISE constraint, otherwise you’ll filter out community capacity.
        #
        # Example (community):
        #   constraints=[]
        #   compute_provider_config_overrides={"cloudType": "COMMUNITY"}
        #
        # compute_provider_config_overrides={"cloudType": "COMMUNITY"},  # Runpod-only
        # Placement / compliance constraints (scheduler-side filtering)
        constraints=[
            DeploymentConstraint.ENTERPRISE,
            DeploymentConstraint.SINGLE_DEVICE,
        ],  # SOC2 compliant T3/T4 datacenters only
        # --- Hardware selection (when applicable) ---
        # Rule: If compute_provider is LOCAL, `num_gpus` and `gpu` do nothing (ignored).
        # For non-local providers, hardware is either auto-selected by Conduit or controlled via
        # provider-specific mechanisms (not via LOCAL-style pinning).
        #
        # num_gpus=2,
        # gpu=GPUS.L4,
        # Replica count (LMLite does round-robin load balancing across replicas)
        replicas=1,
    )

    @dataclass
    class RawHackerNewsHtml:
        raw: str

    @dataclass
    class HackernewsArticle:
        title: str = ""
        points: int = 0
        link: str = ""

    @dataclass
    class HackerNewsSqliteDatabaseSchema:
        sqlite_create_table: str

    @dataclass
    class SchemaWriteOperation:
        file_content: str

    @dataclass
    class SqlLiteArticleListTableInsert:
        sql_command: str

    @dataclass
    class HackernewsArticleList:
        articles: List[HackernewsArticle]

    # File system write block that writes the schema file; all blocks are callable
    write_operation = FileSystemWriteBlock(
        input=SchemaWriteOperation,
        path=Path("schema.sql"),
    )
    while True:
        print("waiting for ready signal...")
        time.sleep(5)
        if lm_lite_block.ready:
            # The sqlite block expects a schema; let the AI write it if needed
            if not os.path.exists("schema.sql"):
                schema_create = lm_lite_block(
                    model_id=model,
                    input=HackernewsArticle,  # INPUT must have data even if it's default data
                    output=HackerNewsSqliteDatabaseSchema,
                    guidance="Output a database table named 'articles' IF NOT EXISTS for the provided data",
                )

                file_content = SchemaWriteOperation(schema_create.sqlite_create_table)
                write_res = write_operation(file_content)

                if write_res.success:
                    print(
                        f"successfully wrote schema: {schema_create.sqlite_create_table}"
                    )
                else:
                    raise RuntimeError(
                        f"failed to write schema file: {write_res.reason}"
                    )

            database_block = Sqlite3Block(
                input=SqlLiteArticleListTableInsert,
                database_url="hackernews.db",
            )

            hacker_news_data = hacker_news_get_block()

            if hacker_news_data.success:
                raw = (
                    hacker_news_data.data
                )  # raw HTML (works with any endpoint: json, etc.)
                if raw:
                    extract_articles_into_list = lm_lite_block(
                        model_id=model,
                        input=RawHackerNewsHtml(raw=raw),
                        output=HackernewsArticleList,
                    )
                    # Output type from previous call is the input to the next call.
                    transform_list_into_insert_command = lm_lite_block(
                        model_id=model,
                        input=extract_articles_into_list,
                        output=SqlLiteArticleListTableInsert,
                    )
                    # Execute the SQL command with the database block
                    db_insert = database_block(transform_list_into_insert_command)
                    if db_insert.success:
                        print(
                            f"successfully saved: "
                            f"{transform_list_into_insert_command.sql_command}"
                        )
                    else:
                        raise RuntimeError(f"DB Insert failed: {db_insert.reason}")

                    lm_lite_block.delete()


if __name__ == "__main__":
    test_hacker_news_scrape()
