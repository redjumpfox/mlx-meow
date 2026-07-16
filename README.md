read [wiki](about:blank) to run.

# mlx-meow

This repo is an aggressive fork of mlx-lm meant for performance. Reason for this fork is slow development and merging of performance based PRs on the upstream.

> Here put a dropdown summary menu with list of differences

This is a strongly opiniated fork;
Current MLX ai slop forks always converge to adding useless bloat such as forced UI, tool calls, and so on that I believe do not have their place in an inference engine.

# Philosophical concept

```
The AI inference engine does not expose tool calls.
Tool calls should be owned by the agentic framework not the inference engine¹.

The AI inference engine does not force behavior unless it's compatibility bound.

Tje AI inference engine is not rigid. Exposed controls can be modified.

¹ The only thing an AI inference may expose in this sense, is an endpoint to its low-level decoding controls to let agentic framework constraints tool calls.
```
# Quick numbers

Faster than most AI slop, for precise benchmark reads the [benchmark page](about:blank) in the wiki.

**simple table:**

 / llama.cpp | MLX | RapidMLX | MTPLX | mlx-meow
Qwen3.6 27B
Qwen3 8B


Add MTP supports to Qwen3.5 models
Qwen3.6 27B , and so on..

I will make a table later.


Faster than MTPLX, rapid-mlx, etc..
strongly scoped on qwen
