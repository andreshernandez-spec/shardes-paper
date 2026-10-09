# Where the time goes: decoding, LoRA members, answer lengths and a gradient, 2026-10-09

ES's progress per hour is set by member-rollouts per second (`../lowrank-check/`). This
measures the rates behind it and what RL pays per token for its gradient.
`es_vllm/throughput_bench.py` at b9720b3 on a Secure Cloud H200 (US-CO-1, driver
580.178.04, 17:12 to 17:52 UTC), about $3.50. Records `throughput.jsonl` and `answers.json`
(every answer's length and reward), logs `pod-log-*.txt.gz`. Numbers:

    python -m es_vllm.throughput_bench --analyze runs/throughput-bench

The Tulu start decoded the 192 iteration-0 prompts 32 times over, greedily (6,144 answers,
cap 2,048).

## Decoding

| engine, requests | sequences in flight | tokens/s | answers/s |
|---|---|---|---|
| plain, base model | 512 | 10,560 | 30.7 |
| plain, base model | 1,024 | 10,720 | 30.9 |
| LoRA engine, base model | 512 | 10,490 | 30.5 |
| LoRA engine, 32 rank-1 members | 512 | 6,970 | 20.7 |
| LoRA engine, 32 rank-1 members | 1,024 | 7,130 | 21.1 |

- Serving members costs a factor of 1.5: a perfect LoRA kernel could raise ES's rate by
  at most that. An engine that has LoRA enabled but serves the base model loses nothing.
- More sequences in flight do not help: the H200 decodes the 8B at about 10,600 tokens/s
  in this setting either way.

## A length cap does not pay

| cap (tokens) | member answers longer | tokens beyond the cap | member reward changes that involve a longer answer |
|---|---|---|---|
| 512 | 24% | 24% | 41% |
| 768 | 11% | 12% | 14% |
| 1,024 | 5.2% | 5.5% | 7.0% |
| 1,536 | 1.5% | 1.4% | 2.4% |

Member answers average 336 tokens and 0.7% reach the 2,048 cap. Every cap gives up at
least as large a share of the reward changes that rank the members as of the tokens it
saves.

## RL's gradient against ES's rollouts

On one H200, without optimizer or communication: a forward and backward pass of the 8B
(gradient checkpointing, micro-batches of 2 as the RL run's) at 4,465 tokens/s, a
forward without gradient (RL's reference model) at 16,400 tokens/s, peak memory 33 GB.
The RL run's sequences average 782 tokens (prompt and answer).

GPU-seconds per 192 prompts at these rates (four RL steps, one ES update at 512 members):

| | GPU-seconds | gain per 192 prompts |
|---|---|---|
| RL: decode 3,072 answers | 100 | |
| RL: gradient on them | 538 | |
| RL: reference forward | 146 | |
| RL total | 784 | +0.040 (held out, first 120 steps) |
| ES, 512 LoRA members | 4,744 | +0.0365 (one update, `../lowrank-check/`) |

- The gradient is 69% of RL's compute and the reference forward another 19%; ES pays
  neither. It pays instead for 32 times as many answers.
- Per unit of gain, ES uses 6.6 times RL's GPU-seconds; with the 64-member dead-prompt
  screen (`lowrank_run.py --screen-check`: 54% of the rollouts) 3.5 times; at the base
  model's decoding rate as well, 2.3 times.
- These are single-GPU rates. RL on a real node adds gradient communication, weight
  broadcasts to the inference engine and generation and training waiting on each other;
  ES's members are independent and exchange only their fitness numbers. Both effects
  favour ES in wall clock, by amounts not measured here.
