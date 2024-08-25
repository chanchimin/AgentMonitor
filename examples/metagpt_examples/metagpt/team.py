#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : team.py
@Modified By: mashenquan, 2023/11/27. Add an archiving operation after completing the project, as specified in
        Section 2.2.3.3 of RFC 135.
"""
import os.path
import warnings
from pathlib import Path
from typing import Any, Optional
import json
import copy

from pydantic import BaseModel, ConfigDict, Field

from metagpt.actions import UserRequirement
from metagpt.const import MESSAGE_ROUTE_TO_ALL, SERDESER_PATH
from metagpt.context import Context
from metagpt.environment import Environment
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.utils.common import (
    NoMoneyException,
    read_json_file,
    serialize_decorator,
    write_json_file,
)


class Team(BaseModel):
    """
    Team: Possesses one or more roles (agents), SOP (Standard Operating Procedures), and a env for instant messaging,
    dedicated to env any multi-agent activity, such as collaboratively writing executable code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Optional[Environment] = None
    investment: float = Field(default=10.0)
    idea: str = Field(default="")

    def __init__(self, context: Context = None, **data: Any):
        super(Team, self).__init__(**data)
        ctx = context or Context()
        if not self.env:
            self.env = Environment(context=ctx)
        else:
            self.env.context = ctx  # The `env` object is allocated by deserialization
        if "roles" in data:
            self.hire(data["roles"])
        if "env_desc" in data:
            self.env.desc = data["env_desc"]

    def serialize(self, stg_path: Path = None):
        stg_path = SERDESER_PATH.joinpath("team") if stg_path is None else stg_path
        team_info_path = stg_path.joinpath("team.json")
        serialized_data = self.model_dump()
        serialized_data["context"] = self.env.context.serialize()

        write_json_file(team_info_path, serialized_data)

    @classmethod
    def deserialize(cls, stg_path: Path, context: Context = None) -> "Team":
        """stg_path = ./storage/team"""
        # recover team_info
        team_info_path = stg_path.joinpath("team.json")
        if not team_info_path.exists():
            raise FileNotFoundError(
                "recover storage meta file `team.json` not exist, " "not to recover and please start a new project."
            )

        team_info: dict = read_json_file(team_info_path)
        ctx = context or Context()
        ctx.deserialize(team_info.pop("context", None))
        team = Team(**team_info, context=ctx)
        return team

    def hire(self, roles: list[Role]):
        """Hire roles to cooperate"""
        self.env.add_roles(roles)

    @property
    def cost_manager(self):
        """Get cost manager"""
        return self.env.context.cost_manager

    def invest(self, investment: float):
        """Invest company. raise NoMoneyException when exceed max_budget."""
        self.investment = investment
        self.cost_manager.max_budget = investment
        logger.info(f"Investment: ${investment}.")

    def _check_balance(self):
        if self.cost_manager.total_cost >= self.cost_manager.max_budget:
            raise NoMoneyException(self.cost_manager.total_cost, f"Insufficient funds: {self.cost_manager.max_budget}")

    def run_project(self, idea, send_to: str = ""):
        """Run a project from publishing user requirement."""
        self.idea = idea

        # Human requirement.
        self.env.publish_message(
            Message(role="Human", content=idea, cause_by=UserRequirement, send_to=send_to or MESSAGE_ROUTE_TO_ALL),
            peekable=False,
        )

    def start_project(self, idea, send_to: str = ""):
        """
        Deprecated: This method will be removed in the future.
        Please use the `run_project` method instead.
        """
        warnings.warn(
            "The 'start_project' method is deprecated and will be removed in the future. "
            "Please use the 'run_project' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_project(idea=idea, send_to=send_to)

    # TODO comment this line for experiment, because it **silently** serialize the process while error occurs, which is
    #  not appropriate while we are doing experiment.
    # @serialize_decorator
    async def run(self, n_round=3, idea="", send_to="", auto_archive=True, **kwargs):
        """Run company until target round or no money"""
        if idea:
            self.run_project(idea=idea, send_to=send_to)

        cur_turn = 0
        task_results = {}
        while n_round > 0:
            n_round -= 1
            cur_turn += 1
            self._check_balance()
            await self.env.run()
            if kwargs["store_intermediate_step"]:
                if kwargs["output_path_prefix"][-1] == "/":
                    cur_store_path = kwargs["output_path_prefix"][:-1] + f"_turn_{cur_turn}"
                else:
                    cur_store_path = kwargs["output_path_prefix"] + f"_turn_{cur_turn}"
                kwargs["monitor"].recording(os.path.join(cur_store_path, kwargs["output_path_postfix"]),
                                            task_instruction=kwargs["task_instruction"],
                                            task_trajectory=self.env.history,)

            # store the code, test, answer here, we have first check that different architecture contains overall the same field
            meta_information = {}
            for key, value in kwargs.items():
                if "get_global_" in key:
                    meta_information[key] = copy.deepcopy(value())
            # meta_information = {
            #     "code": kwargs["get_global_code"](),
            #     "test": kwargs["get_global_test"](),
            #     "answer": kwargs["get_global_answer"](),
            #     "response": kwargs["get_global_response"](),
            # }
            task_results[f"turn_{cur_turn}"] = meta_information
            logger.debug(f"max {n_round=} left.")

        self.env.archive(auto_archive)
        pretty_data = json.dumps(task_results, indent=4)
        print(pretty_data)
        return self.env.history, task_results
