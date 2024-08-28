import asyncio
import json
import logging
import os
import traceback
import inspect
import io
import sys
from typing import Type, Dict, Any, List
from pydantic import BaseModel, Field

from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode.streaming.action.base_action import BaseAction


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RunPythonActionConfig(ActionConfig, type=ActionType.RUN_PYTHON):
    starting_phrase: str


class RunPythonParameters(BaseModel):
    code: str = Field(..., description="The Python code to be executed")
    params: dict = Field(
        ..., description="Parameters for the Python function as a DICT-encoded string"
    )


class RunPythonResponse(BaseModel):
    status: str = Field(..., description="The status of the Python code execution")
    response: Dict[str, Any] = Field(
        ..., description="The response from the Python code"
    )


class RunPython(
    BaseAction[RunPythonActionConfig, RunPythonParameters, RunPythonResponse]
):
    description: str = "Executes a Python function and returns the response"
    parameters_type: Type[RunPythonParameters] = RunPythonParameters
    response_type: Type[RunPythonResponse] = RunPythonResponse

    def code_runner(self, func_or_string, input_dict):
        output = {"result": None, "error": None, "metadata": {}, "stdout": ""}

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            if isinstance(func_or_string, str):
                local_namespace = {}
                exec(func_or_string, globals(), local_namespace)
                func_name = next(
                    (name for name, obj in local_namespace.items() if callable(obj)),
                    None,
                )
                if func_name is None:
                    raise ValueError("No function found in the provided string")
                func = local_namespace[func_name]
            else:
                func = func_or_string

            sig = inspect.signature(func)
            params = sig.parameters

            missing_args = [
                param
                for param in params
                if param not in input_dict
                and params[param].default == inspect.Parameter.empty
            ]
            if missing_args:
                raise ValueError(
                    f"Missing required arguments: {', '.join(missing_args)}"
                )

            valid_args = {k: v for k, v in input_dict.items() if k in params}

            for param, value in valid_args.items():
                if params[param].annotation != inspect.Parameter.empty:
                    if not isinstance(value, params[param].annotation):
                        raise TypeError(
                            f"Argument '{param}' should be of type {params[param].annotation.__name__}, but got {type(value).__name__}"
                        )

            result = func(**valid_args)
            output["result"] = result

            output["metadata"]["function_name"] = func.__name__
            output["metadata"]["args_received"] = valid_args
            output["metadata"]["return_type"] = type(result).__name__

        except Exception as e:
            output["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
        finally:
            output["stdout"] = sys.stdout.getvalue()
            sys.stdout = old_stdout

        return output

    async def run(
        self, action_input: ActionInput[RunPythonParameters]
    ) -> ActionOutput[RunPythonResponse]:
        logger.debug(f"Action input: {action_input}")

        code = action_input.params.code
        params = action_input.params.params

        response_content = self.code_runner(code, params)
        logger.debug(f"Response: {response_content}")

        status = "success" if response_content["error"] is None else "error"

        return ActionOutput(
            action_type=action_input.action_config.type,
            response=RunPythonResponse(status=status, response=response_content),
        )
