# Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   This installs SUMO 1.27.1 itself (`eclipse-sumo`) as a pip package, not
   just its Python bindings — no system package (`apt install sumo`) needed.
   (Ubuntu's `apt` package is version 1.18, which does not match what
   `truck_env` is written against.)

2. Set `SUMO_HOME` to the pip-installed package directory before running
   anything that imports `truck_env` (it checks for this at import time):

   ```
   export SUMO_HOME=$(python3 -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
   export PATH=$SUMO_HOME/bin:$PATH
   ```

   Put these two lines in your shell profile or venv `activate` script so you
   don't have to repeat them every session.

3. Verify the env works:

   ```
   python3 examples/smoke_test.py
   ```

   Expect an episode line (a random policy usually ends the episode in a few
   steps via `outside_road` or similar) followed by
   `OK: 100/100 resets succeeded.`

## Neurosymbolic scripts (optional)

`truck_env/neurosymbolic_*.py` and `examples/train_neurosymbolic*.py` also
import `scallopy`, which is **not** in `requirements.txt` — PyPI has no Linux
wheel for it. To use those scripts, build it from source:

```
git clone https://github.com/scallop-lang/scallop
cd scallop/etc/scallopy
pip install maturin
maturin develop --release
```

This has not been verified in this environment.
