# 02. Tulu 3.1's data path, read from open-instruct at 3f37c29

Work in progress (T0 of `01-tulu31-plan.md`), started 2026-09-26. Read from
`open_instruct/grpo_vllm_thread_ray_gtrl.py` and `open_instruct/dataset_transformation.py`
at 3f37c29, the commit the Tulu 3.1 model card names, with the card's command.

## What the code does

1. **Data.** `get_cached_dataset_rlvr(["allenai/RLVR-GSM-MATH-IF-Mixed-Constraints",
   "1.0"], ["train"], tc, ...)`. A fraction of 1.0 gives `update_range(len(dataset))`,
   which is `select(range(n))`: the whole set, in file order. No sampling.
2. **Transforms.** `rlvr_tokenize_v1` renders the prompt (the messages, or all but the
   last) with the chat template and `add_generation_prompt=True`, and the full messages
   without it; `rlvr_filter_v1` keeps rows whose prompt and full sequence fit their limits.
3. **Two quirks, both harmless for this run but reproduced as they are:**
   - The caller passes `max_prompt_token_length` and `max_token_length` in each other's
     positions. Both are 2,048 in the card's command, so nothing changes.
   - `train_dataset.shuffle(seed=args.seed)` discards its result: `Dataset.shuffle`
     returns a new dataset. The training set therefore stays in file order (after the
     filter), and the only randomness in which prompts a step sees is the iterator below.
4. **Prompts per step.** `rollout_batch_size = local_rollout_batch_size x world_size`,
   with `world_size = sum(actor_num_gpus_per_node)`; the card's command gives 8 x 6 = 48,
   matching the card's "48 * 16 = 768" rollouts per step.
5. **Order.** `ShufflingIterator(np.arange(n), 48, seed=1)`: one shuffle of the indices
   with `default_rng(1)`, batches of 48, the remainder dropped, a reshuffle from the same
   generator at each epoch boundary. Same logic as the Olmo 3 run's iterator, without the
   exclusion list.
6. **Rewards for responses without eos.** `--non_stop_penalty` replaces their score with
   `penalty_reward_value`, 0.0 in the card's command: a response that runs to the 2,048
   cap scores 0.
7. **Tokenizer.** `TokenizerConfig(chat_template_name="tulu", add_bos=False)` (the
   default; the command has no `--add_bos`), built by `get_tokenizer_tulu_v1`. The
   `tulu` template has no bos token, so prompts carried none unless that function adds
   one (to check).

## To do

- Check `get_tokenizer_tulu_v1` for anything that changes the template or adds tokens.
- Rebuild the filtered set and the prompt stream for steps 1 to 2,440 (the last branch);
  record them as for Olmo 3, with the count the filter keeps.
- Vendor the verifiers (GSM8K, MATH flex, IF) with tests.
