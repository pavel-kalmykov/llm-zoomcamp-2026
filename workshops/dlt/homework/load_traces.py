import os

import dlt
import requests
from dotenv import load_dotenv

load_dotenv()

LOGFIRE_READ_TOKEN = os.environ["LOGFIRE_READ_TOKEN"]
LOGFIRE_REGION = os.environ.get("LOGFIRE_REGION", "us")
QUERY_URL = f"https://logfire-{LOGFIRE_REGION}.pydantic.dev/v1/query"


def fetch_agent_traces():
    sql = "SELECT * FROM records ORDER BY start_timestamp"
    response = requests.get(
        QUERY_URL,
        headers={"Authorization": f"Bearer {LOGFIRE_READ_TOKEN}"},
        params={"sql": sql},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    columns = {c["name"]: c["values"] for c in data["columns"]}
    n_rows = len(next(iter(columns.values())))
    for i in range(n_rows):
        yield {name: values[i] for name, values in columns.items()}


@dlt.resource(name="records")
def agent_traces():
    yield from fetch_agent_traces()


def main():
    pipeline = dlt.pipeline(
        pipeline_name="agent_traces_pipeline",
        destination=dlt.destinations.duckdb("traces_db.duckdb"),
        dataset_name="agent_traces",
    )
    load_info = pipeline.run(agent_traces())
    print(load_info)


if __name__ == "__main__":
    main()
