# Tulu 3.1 verifiers, vendored from open-instruct at 3f37c29

`if_functions.py`, `math_utils.py` and `ground_truth_utils.py` are
[allenai/open-instruct](https://github.com/allenai/open-instruct) files at commit
`3f37c29ddc97d2c108a7658692d2d2c3708ef182`, the commit the Tulu 3.1 model card names,
under the Apache License 2.0 (`LICENSE-open-instruct`). The only change: the two
`from open_instruct.<module> import` lines in `ground_truth_utils.py` import from this
package instead. sha256 of the originals:

    ec9b8d402ac5032470157afaebc55bbf65b1094b5a93778946fd21c2db0a25c4  if_functions.py
    06106b69de337701093053807149d245dbdbc2f1cf82f102d1e72c7a171e2d88  math_utils.py
    324f4a6b730b7d0214924350b2fab863b5ed326b13d4cb859c95b8bf25d0c671  ground_truth_utils.py

`__init__.py` is ours: the per-response form of `model_utils.apply_verifiable_reward`.

Environment. open-instruct at 3f37c29 pinned `antlr4-python3-runtime==4.11.0` and
`sympy==1.13.1` (its `requirements.txt`). The MATH verifier parses LaTeX with sympy,
and `math_utils.is_equiv` returns False on an ImportError, so without antlr a MATH
answer that needs parsing scores 0 and nothing says so. `check_environment()` raises
instead. Whether a newer sympy (as torch requires) changes any verdict is not yet
checked; the fallback is a small separate environment for the verifier.
