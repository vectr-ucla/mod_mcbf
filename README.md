# Optimal-decay matrix control barrier functions

#### [[ArXiv](https://arxiv.org/abs/2610.18675)] [[Code](https://github.com/vectr-ucla/mod_mcbf)]

Code for the paper *Optimal Decay Matrix Control Barrier Functions: Soundness and
Completeness for Matrix Safety Constraints* by Samuel G. Gessow, Albert Wu,
Aaron D. Ames and Brett T. Lopez.

A matrix control barrier function (MCBF) enforces a constraint `H(x) >= 0` on a
symmetric matrix. The paper compares three safety filters:

```
standard MCBF   Hdot + alpha(H)                         >= 0
scalar OD       Hdot + omega * alpha(H)                 >= 0,   omega >= 0
matrix OD       Hdot + Omega' alpha(H) + alpha(H) Omega >= 0
```

The standard and scalar filters cannot certify motions that keep the spectrum of `H`
fixed while coupling its kernel to its positive eigenspace. Matrix OD can, and the
filter is still a convex program.

The example in Section VIII of the paper is singularity avoidance for a planar
two-link arm, with `H = J J^T - sigma_bar^2 I` in absolute joint angles. This
repository contains the filters, the arm model, and the scripts that produce the
figures and numbers in that section.

The repository also has a version of the example in relative joint angles, which
was cut from the paper for space. See
[Relative joint angles](#relative-joint-angles-not-in-the-paper).

## Setup

Python 3.12 or newer. On 3.10 or 3.11, install from `requirements.txt` instead of the
lock file.

```sh
git clone https://github.com/vectr-ucla/mod_mcbf.git
cd mod_mcbf
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
```

The scripts import `odmcbf` from the repository root, so the package does not need to
be installed. To run the notebooks, start Jupyter from the repository root with the
environment active, and pick the default "Python 3" kernel:

```sh
jupyter lab main_relative.ipynb
```

Saving the notebook animation as an mp4 needs [FFmpeg](https://ffmpeg.org)
(`brew install ffmpeg` or `sudo apt install ffmpeg`). Everything else runs without it.

## Reproducing the paper

```sh
python scripts/reproduce_paper.py
```

This takes about 30 s. It runs the experiments defined in `odmcbf/paper.py`, prints
the numbers quoted in Section VIII, and writes the figures to `figures/paper/` with
the file names the paper uses. `--copy-to DIR` also copies them into the paper's
figure directory.

| Paper | Experiment in `odmcbf/paper.py` | Standalone script |
|---|---|---|
| Fig. 1(a), (b) | `hero` | `scripts/closed_loop_demo.py` (default case) |
| Fig. 2, Sec. VIII-C | `tracking` | `scripts/tracking_figure.py` |
| Fig. 3 | `recovery` | `scripts/recovery_demo.py` |

`odmcbf/tests/test_paper.py` checks the closed forms in Sec. VIII-A and VIII-B
(eqs. (27) to (37)) and the numbers quoted in the section.

The results are deterministic except for solve times. We produced them with the
versions in `requirements-lock.txt` (Python 3.12.3, numpy 2.5.3, cvxpy 1.9.3 with
Clarabel 0.11.1, matplotlib 3.11.2). Other cvxpy versions give the same numbers, but
the figure text depends on the matplotlib version. If MOSEK is installed the filters
use it instead of Clarabel; we have not rerun the paper's experiments with MOSEK.

## Repository layout

```
odmcbf/
  filters.py            standard MCBF, scalar OD, and matrix OD safety filters
  two_link.py           two-link arm and barrier H, absolute joint angles (paper)
  two_link_relative.py  same arm in relative joint angles (not in the paper)
  simulate.py           closed-loop rollouts, including recovery from outside the safe set
  tracking.py           moving target and the nominal tracking controller, eq. (32)
  regions.py            where H is not positive definite, in joint space
  figures.py            plotting and animation
  paper.py              the Section VIII experiments
  tests/                unit tests and checks against the paper
scripts/                command-line scripts, listed below
main.ipynb              constant-command example in absolute angles (not in the paper)
main_relative.ipynb     constant-command example in relative angles (not in the paper)
two_link_manipulator_kinematics.md   derivations for the two-link arm
```

`two_link_manipulator_kinematics.md` derives `H`, the standard MCBF's cap on the
rotation rate about joint 1, and the configurations where scalar and matrix OD differ.
It uses absolute joint angles `theta = (theta1, theta2)`, as the paper does.

## Relative joint angles (not in the paper)

The paper writes the arm in absolute joint angles, `theta1` and `theta2`, both measured
from the horizontal. The alternative is relative angles `q = (q1, q2)`, where `q1` is
the base angle and `q2` is the elbow angle measured from the first link, so the link
angles are `(q1, q1 + q2)`. In these coordinates `J`, `H` and the filters' control
cost are all in terms of the motor rates `qdot`, which puts a different metric on
joint velocities than the paper uses. We ran the example in both coordinate systems but only had
room for the absolute-angle version in the paper.

`odmcbf/two_link_relative.py` defines `RelativeTwoLinkArm`, which has unit links and
`sigma_bar^2 = (3 - sqrt 5)/2`, placing the boundary at `q2 = pi/2`. Because `q1` only
rotates `H`, the safety margin depends on `q2` alone. The class has the part of
`TwoLinkArm`'s interface that `SafetyFilter`, `run_comparison` and the plotting code use,
so they work with it unchanged.

`main_relative.ipynb` runs the closed loop. The arm starts at `q = (0, 2 pi/3)`
under a constant command `u = (1, -0.6)` rad/s for 8 s, so the elbow extends toward
the boundary while the base turns. At the boundary only the elbow needs to stop; the
base can keep turning at 1 rad/s without changing the margin. Out of the commanded
8 rad of base rotation, the standard MCBF completes 1.02 rad, scalar OD 3.12 rad and
matrix OD 7.91 rad. `test_two_link_relative.py` checks these numbers. The notebook
also animates the run and, if FFmpeg is installed, saves the video as
`figures/two_link_velocity_tracking.mp4`.

`main.ipynb` runs the same kind of constant-command experiment on the paper's
absolute-angle arm.

## Other scripts

All scripts accept `--help`, and most take `--l1`, `--l2` and `--sigma-bar` to change
the arm. Figures go to `figures/`.

- `closed_loop_demo.py` runs the three filters while the end effector tracks a target
  that enters a violated band. With no options it is the paper's Fig. 1 run.
  `--case inner|outer|graze|rotate` picks other targets and `--video FILE` saves an
  animation.
- `tracking_figure.py` produces Fig. 2 from six tracking cases.
- `recovery_demo.py` starts the arm inside a violated band and runs matrix OD with the
  recovery constraint of Section VII. The margin recovers at rate `2 beta gamma`.
- `certified_velocities_demo.py` plots the smallest scalar decay that certifies each
  joint velocity at the boundary (Proposition 2), the plot of Fig. 1(b) for any arm,
  and checks the closed forms numerically.
- `workspace_demo.py` draws the arm in its workspace with the violated bands shaded.
- `not_pd_regions_demo.py` plots where `H` is not positive definite in
  `(theta1, theta2)`.

```sh
python scripts/closed_loop_demo.py --case inner --l1 1 --l2 0.6 --sigma-bar 0.3
python scripts/recovery_demo.py --band inner
python scripts/workspace_demo.py --l1 1.5 --l2 0.8 --sigma-bar 0.78
```

## Using the package

```python
import numpy as np
from odmcbf import TwoLinkArm

arm = TwoLinkArm(1.0, 0.75, sigma_bar=0.6)       # any 0 < sigma_bar < min(l1, l2)
arm = TwoLinkArm.from_boundary(np.pi / 3, 1, 1)  # or place the boundary directly
```

The three filters are modes of `odmcbf.SafetyFilter`, and `odmcbf.rollout` simulates
a closed loop with any of them. `odmcbf/paper.py` shows how the pieces fit together.

## Tests

```sh
pytest          # about 2100 tests, a few minutes
pytest --cov    # with coverage (100%)
```

Besides one test file per module, four files check the code against the paper.
`test_manuscript.py` covers Theorems 1 to 4, Propositions 1 to 3 and Table I, on the
arm and on synthetic examples. `test_manuscript_example.py` checks the Section VIII
closed forms on other arms, `test_recovery.py` checks the Section VII recovery rate in
closed loop, and `test_paper.py` checks the Section VIII numbers.

Some parametrized tests are skipped where they do not apply. `SafetyFilter` raises
`FilterInfeasible` when the solver reports a non-optimal status. When the solver
fails outright, as Clarabel can on the weakly infeasible problems at the boundary,
cvxpy's `SolverError` passes through unchanged.

CI (`.github/workflows/tests.yml`) runs the tests with the lock file, with the
minimum versions in `requirements.txt`, and with the latest releases.

## Citation

If you found this work useful, please cite our manuscript:

```bibtex
@article{Gessow26:ODMCBF,
  author={Gessow, Samuel G. and Wu, Albert and Ames, Aaron D. and Lopez, Brett T.},
  journal={arXiv preprint arXiv:2610.18675},
  title={Optimal Decay Matrix Control Barrier Functions: Soundness and Completeness for Matrix Safety Constraints},
  year={2026}
}
```
