"""Adapters for admitting and smoke-testing the CoRL / ULM-R1 GRPO stack.

This package is scoped to task T710. It intentionally keeps all heavy
(torch/trl/corl/janus) imports inside function bodies of ``run_smoke`` so
that the pure-python pieces (``param_policy``, ``paths``, ``micro_split``)
remain importable -- and unit-testable -- in a plain CPU-only dev
environment that does not have torch installed.
"""
