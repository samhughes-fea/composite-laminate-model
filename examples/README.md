# Examples

Worked examples of the laminate model on its own: no mesh and no solver. Each one
lives in its own folder with its script, the figures it produces and a README that
walks through the model and how to read the results.

| # | Example | What it shows | Main API |
|---|---|---|---|
| 01 | [Failure envelopes](01_failure_envelopes/) | Maximum stress, Tsai-Hill, Tsai-Wu and Hashin envelopes for a T300/5208 ply | failure criteria |
| 02 | [Stiffness polars](02_stiffness_polars/) | Ex, Gxy and νxy against direction for four stacks | `Laminate`, `integrate_abd` |

Examples that solve plates, shells and blades (kinematic models against Navier,
through-thickness stresses, progressive damage, blade beam properties and more)
are in [shell-fem](https://github.com/samhughes-fea/shell-fem/tree/main/examples).
The beam-section examples are in
[anisotropic-section-model](https://github.com/samhughes-fea/anisotropic-section-model/tree/main/examples).

## Running

From the repository root:

```bash
pip install -e ".[examples]"     # the package plus matplotlib
python examples/run_all.py       # every example, figures into each folder
```

Figures are committed so the READMEs render on GitHub.
