from dataclasses import dataclass
from typing import List
import os
from pathlib import Path

from conduit.conduit_types import GPUS
from conduit.runtime import LMLiteBlock
from conduit import ModelConfig, ComputeProvider
from conduit.blocks import FileSystemWriteBlock, HttpGetBlock, Sqlite3Block


def test_hacker_news_scrape(model: str = "Qwen/Qwen3-4B-Instruct-2507") -> None:
    """
    Scrapes the Hacker News front page and stores parsed articles into a SQLite DB using LMLite.
    """

    # Callable that returns an HTTP operation
    hacker_news_get_block = HttpGetBlock(endpoint="https://news.ycombinator.com/")

    # Runtime block, LMLite, an AI inference engine
    lm_lite_block = LMLiteBlock(
        models=[
            # add as many models as you want! Conduit will calculate model and GPU reqs or throw an error
            ModelConfig(
                model,  # HF model id
                max_model_len=262144,  # set to examples max
                max_model_concurrency=1,  # batch size pool
                model_batch_execute_timeout_ms=1000,
            ),
        ],
        compute_provider=ComputeProvider.RUNPOD,  # choose compute provider (only RUNPOD atm)
        # choose a specific GPU OR by default Conduit will optimize for VRAM efficiency
        gpu=GPUS.NVIDIA_L40,
        # default = 1. LMLite uses round-robin load balancing across replicas.
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

    if lm_lite_block.ready:
        # The sqlite block expects a schema; let the AI write it if needed
        if not os.path.exists("schema.sql"):
            schema_create = lm_lite_block(
                model_id=model,
                input=HackernewsArticle,  # INPUT must have data even if it's default data
                output=HackerNewsSqliteDatabaseSchema,
                guidance="Output a database table named 'articles' for the provided data",
            )

            file_content = SchemaWriteOperation(schema_create.sqlite_create_table)
            write_res = write_operation(file_content)

            if write_res.success:
                print(f"successfully wrote schema: {schema_create.sqlite_create_table}")
            else:
                raise RuntimeError(f"failed to write schema file: {write_res.reason}")

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


if __name__ == "__main__":
    test_hacker_news_scrape()
