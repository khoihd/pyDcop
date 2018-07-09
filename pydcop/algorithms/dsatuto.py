#!/usr/bin/env python3

# BSD-3-Clause License
#
# Copyright 2017 Orange
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""

This module contains a very simple implementation of DSA, for demonstration
purpose.

To keep things as simple as possible, we implemented the bare minimum,
and avoided some details you would generally care about:

* no algorithm parameters (threshold, variants, etc.)
* no computation footprint nor message size

"""


import logging
from typing import Any, Tuple

from numpy import random

from pydcop.algorithms import ComputationDef, assignment_cost
from pydcop.infrastructure.computations import VariableComputation, message_type

# Type of computations graph that must be used with dsa
GRAPH_TYPE = 'constraints_hypergraph'


# No parameters for our simplistic DSA implementation
algo_params = []


def build_computation(comp_def: ComputationDef):
    return DsaTutoComputation(comp_def.node.variable,
                              comp_def.node.constraints,
                              computation_definition= comp_def)


DsaMessage = message_type("DsaMessage", ["value"])


class DsaTutoComputation(VariableComputation):
    """
    A very simple DSA implementation.

    """
    def __init__(self, variable, constraints, computation_definition):
        super().__init__(variable, computation_definition)

        self._msg_handlers['DsaMessage'] = self.on_value_msg
        self.logger = logging.getLogger('pydcop.algo.dsatuto.'+variable.name)
        self.constraints = constraints
        self.current_cycle = {}
        self.next_cycle = {}

    def on_start(self):
        self.random_value_selection()
        self.logger.debug(
            "Random value selected at startup : %s ", self.current_value)
        self.post_to_all_neighbors(DsaMessage(self.current_value))
        self.evaluate_cycle()

    def on_value_msg(self, variable_name, recv_msg, t):

        if variable_name not in self.current_cycle:
            self.current_cycle[variable_name] = recv_msg.value
            self.evaluate_cycle()

        else:  # The message for the next cycle
            self.next_cycle[variable_name] = recv_msg.value

    def evaluate_cycle(self):

        if len(self.current_cycle) == len(self.neighbors):

            self.current_cycle[self.variable.name] = self.current_value
            current_cost = assignment_cost(self.current_cycle, self.constraints)
            arg_min, min_cost = self.compute_best_value()

            self.logger.debug(
                "Evaluate cycle %s: current cost %s - best cost %s",
                self.cycle_count, current_cost, min_cost)

            if current_cost - min_cost > 0 and 0.5 > random.random():
                self.value_selection(arg_min)
                self.logger.debug(
                    "Select new value %s for better cost %s ",
                    self.cycle_count, min_cost)
            else:
                self.logger.debug(
                    "Do not change value %s ", self.current_value)

            self.new_cycle()
            self.current_cycle, self.next_cycle = self.next_cycle, {}
            self.post_to_all_neighbors(DsaMessage(self.current_value))

    def compute_best_value(self) -> Tuple[Any, float]:

        arg_min, min_cost = None, float('inf')
        for value in self.variable.domain:
            self.current_cycle[self.variable.name] = value
            cost = assignment_cost(self.current_cycle, self.constraints)
            if cost < min_cost:
                min_cost, arg_min = cost, value
        return arg_min, min_cost


