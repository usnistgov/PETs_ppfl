# CLAUDE.md

Guidance for working in this repository. Read this before making changes.

> This file is durable, repo-wide context. Keep it short and stable. Volatile facts
> (release dates, current owners, open decisions) belong in issues or a decision log,
> not here. This is guidance, not a guarantee — anything that *must* happen is
> enforced by tests, hooks, or permissions, not by this document.

---

## What this system is

A Python demonstration of a **privacy-preserving federated learning (FL)** system
that models **phenotype from genotype**. It serves two purposes:

1. A **reference architecture** for differentially private (DP) federated learning.
2. A **platform for rapid experimentation** on the privacy/utility trade-offs in such systems.

Every change should protect both roles: keep the architecture clean and readable as a
reference, and keep the experiment path easy to extend.

---

## ⚠️ Privacy-critical invariants — read before touching ML code

The differential privacy guarantee is the entire point of this system. Much of the code
that upholds it is **not protected by functional tests** — a change can pass every test
and still silently invalidate the DP claim. Treat the following as a protected zone.

**Protected modules / paths:**
- `./PETs_Testbed/model.py`

**Rules for any change touching these:**
- Use plan mode and get human review before implementing. Do **not** auto-apply edits here.
- Never alter noise calibration, gradient clipping bounds, the privacy accountant, or
  composition/budget bookkeeping as a side effect of an unrelated change or "cleanup."
- Never introduce a data path that moves raw client data or un-noised gradients across
  the client boundary.
- When proposing a change, **explicitly state how the (ε, δ) guarantee is preserved**,
  including how noise scale relates to sensitivity and how the budget is accounted.
- "It passed the tests and the diff looks reasonable" is **not** sufficient sign-off here.

**Key DP parameters and where they live:**

`./configs/`


---

## Commands


- Run tests: `./PETs_Testbed/tests`


Always run tests, lint, and type-check before considering a change complete.

---

## Coding conventions

<!-- FILL IN / adjust -->
- Python version: `3.10`
- Style: follow existing patterns in the module you're editing; match surrounding code.
- Type hints on all new/edited public functions.
- Dependencies: `./requirements.txt`. Do not add a new dependency
  without flagging it first.
- Prefer small, composable functions over large ones, especially in the privacy layer
  where each step must be auditable.

---

## Testing philosophy

- Tests are the anchor of the workflow — changes should be verified against a red/green cycle.
- Before changing behavior in an untested area, add **characterization tests** first.
- Beyond functional tests, assert on **privacy mechanics** where possible: budget
  accounting, noise scale as a function of (ε, δ) and sensitivity, and clipping bounds.
- Don't weaken or delete a test to make a change pass. If a test is wrong, fix it
  deliberately and explain why.

---

## Workflow expectations

- **Explore → Plan → Implement → Verify → Commit.** Read relevant files first; propose a
  plan before non-trivial work (roughly 3+ steps or any architectural/privacy decision).
- Work on feature branches. **Never commit to `main`.**
- Keep changes small and reviewable — this is a shared repo and teammates must be able to
  reason about every diff. Prefer several focused PRs over one large one.
- At handoff, provide evidence: what changed, why, and test/benchmark results.

---

## Performance work

- Measure, don't guess. Profile a representative experiment run and propose changes with
  **before/after benchmarks**.
- Expect the real costs in FL to sit in serialization, aggregation, and per-round
  communication rather than raw compute — look there first.

---

## Experiment platform

The system exists partly to make new trade-off experiments fast to define and run.
When adding features, keep the experiment-definition path clean and config-driven, and
preserve clear extension points so new experiments don't require touching core code.

 The configuration files to define an experiment are in `./configs/`

---

## Things to avoid

- Do not modify privacy-critical code as an incidental part of an unrelated task.
- Do not add dependencies, change the DP parameters, or alter the client boundary without
  explicit human sign-off.
- Do not make claims about code you haven't read — open the file first.
- Do not process real or sensitive data; this repo is for demonstration and experiments.

---

## Acknowledge AI's role

My organization, the National Institute of Standards and Technology, requires disclosure of Gen AI where it is used. Anytime you modify a file add a breif statement characterizing your role. Include breif model details. Here'a an example: 

"This software was edited with the assistance of Gitlab Duo, developed by Gitlab. Duo was used to write unit test code, perform functional code review,  and suggest functions in accordance with the authors’ instructions. All content, scientific claims, and conclusions have been reviewed and verified by the authors to ensure accuracy and originality."


