#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : MG StanfordTown Env

from metagpt.environment.base_env import Environment
from metagpt.environment.stanford_town.stanford_town_ext_env import StanfordTownExtEnv


class StanfordTownEnv(StanfordTownExtEnv, Environment):
    pass
