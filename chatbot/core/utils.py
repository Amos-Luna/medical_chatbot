import json
import logging
import tempfile
from typing import List

import pandas as pd
from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def extract_metadata(genie_data: dict) -> tuple:
    """Extract usefull metadata"""
    logging.info(f"Extracting metadata from Genie data: {genie_data}")
    try:
        # num_rows = genie_data["statement_response"]["manifest"].get("total_row_count", "")
        schema = genie_data["statement_response"]["manifest"]["schema"].get("columns", [])
        column_names = [col["name"] for col in schema]
        data_array = genie_data["statement_response"]["result"].get("data_array", [[]])
        return (column_names, data_array)
    except Exception:
        logging.error("Error extracting metadata")
        return None


def json_to_dataframe(column_names: List[str], data_array: List[list]) -> pd.DataFrame:
    """Transform metadata to dataframe structure"""
    try:
        df = pd.DataFrame(data_array, columns=column_names)
        return df
    except Exception:
        logging.info("Error generating Dataframe...")


def save_dataframe_temp(df: pd.DataFrame) -> str:
    """Save dataframe and get its temporary path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(temp_file.name, index=False)
    return temp_file.name
