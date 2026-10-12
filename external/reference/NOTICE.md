# LongMemEval reference

The evaluation source is downloaded from the official repository at the exact
commit in configs/checkpoints.lock.json. Only get_anscheck_prompt is compiled;
the source file is not executed as a program and its hosted inference code is
never imported by the active runner.

Upstream: https://github.com/xiaowu0162/LongMemEval
License: MIT, see LICENSE.LongMemEval.
Model substitution: Gemini 3.1 Flash-Lite replaces the original GPT-4o judge.
The task-specific prompt is otherwise taken from the pinned official function.
