from __future__ import annotations

import pytest

from graph_looper.spec import load_graph_str

LINEAR = """
name: linear
nodes:
  - id: start
    type: input
  - id: step
    type: agent
    prompt: "do {{ task }}"
  - id: done
    type: output
edges:
  - from: start
    to: step
  - from: step
    to: done
"""

LOOP = """
name: loop
defaults:
  max_visits: 2
nodes:
  - id: start
    type: input
  - id: worker
    type: agent
    mode: resident
    join: any
    prompt: "attempt {{ iteration }} of {{ task }} — notes: {{ results.check }}"
  - id: check
    type: gate
    choices: [pass, fail]
    on_exhausted: pass
    prompt: "judge {{ inputs.worker }}"
  - id: done
    type: output
edges:
  - from: start
    to: worker
  - from: worker
    to: check
  - from: check
    to: worker
    when: fail
  - from: check
    to: done
    when: pass
"""

FANOUT = """
name: fanout
nodes:
  - id: start
    type: input
  - id: a
    type: agent
    prompt: "a: {{ task }}"
  - id: b
    type: agent
    prompt: "b: {{ task }}"
  - id: merge
    type: transform
    op: concat
    separator: " + "
  - id: done
    type: output
    prompt: "{{ inputs.merge }}"
edges:
  - from: start
    to: a
  - from: start
    to: b
  - from: a
    to: merge
  - from: b
    to: merge
  - from: merge
    to: done
"""


@pytest.fixture
def linear_graph():
    return load_graph_str(LINEAR)


@pytest.fixture
def loop_graph():
    return load_graph_str(LOOP)


@pytest.fixture
def fanout_graph():
    return load_graph_str(FANOUT)
