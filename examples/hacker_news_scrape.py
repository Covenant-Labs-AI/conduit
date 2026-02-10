import time
from dataclasses import dataclass
from typing import List
import os
from pathlib import Path

from conduit.compute_provider.local import LocalNetworkBinding
from conduit.runtime import LMLiteBlock
from conduit import LmLiteModelConfig, ComputeProvider
from conduit.blocks import FileSystemWriteBlock, HttpGetBlock, Sqlite3Block, DbAction
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
#   5) Uses the model again to transform the structured article list into a safe,
#      structured DbAction (no raw SQL), then executes it with Sqlite3Block into
#      hackernews.db.
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
        # Placement / compliance constraints (scheduler-side filtering)
        constraints=[
            DeploymentConstraint.ENTERPRISE,
            DeploymentConstraint.SINGLE_DEVICE,
        ],  # SOC2 compliant T3/T4 datacenters only
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

    # NEW: safe structured DB action wrapper for the new Sqlite3Block API
    @dataclass
    class SqliteArticleDbAction:
        db_action: DbAction

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

            # NEW: Sqlite3Block now loads schema.sql and only accepts structured DbAction
            database_block = Sqlite3Block(
                input=SqliteArticleDbAction,
                database_url="hackernews.db",
                schema_file="schema.sql",
                allowed_tables={"articles"},
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

                    # IMPORTANT: When generating db ops with LLMS there's no no standard yet for output types
                    # out must corherse list of lists as the inert many type. Not ideal but in future will be updated.
                    transform_list_into_db_action = lm_lite_block(
                        model_id=model,
                        input=extract_articles_into_list,
                        output=SqliteArticleDbAction,
                        guidance=(
                            "Create a DbAction that inserts these articles into the SQLite table "
                            "'articles'. Use op='insert_many' with columns ['title','points','link'] "
                            "and rows rows are a list of lists as the article values in the same order."
                        ),
                    )

                    # Execute the DB action with the database block
                    db_insert = database_block(transform_list_into_db_action)
                    if db_insert.success:
                        print(
                            f"successfully saved: rowcount={db_insert.rowcount} "
                            f"lastrowid={db_insert.lastrowid}"
                        )
                    else:
                        raise RuntimeError(f"DB Insert failed: {db_insert.reason}")

                    lm_lite_block.delete()


if __name__ == "__main__":
    test_hacker_news_scrape()
