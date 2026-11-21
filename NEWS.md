# NEWS

## 0.3.5

- Docs: package-name section titles and left-aligned logos (no code changes).
## 0.3.4

- The multistart banner is logged once per session (`maxlog = 1`).
## 0.3.3

- Added docstrings for `HyperoptAlg` and the starting-point samplers (REPL help and API reference).


## 0.3.2

- Fixed the error handling and `x0` aliasing; scheduled formatter workflow.

## 0.3.0

- NonconvexCore 2 with forwarded sub-solver capability declarations;
  DocumenterFragments docs; tests migrated to TestItems with
  JET/Aqua/formatting quality gates; removed the duplicate `HyperoptOptions`
  docs block.

## 0.2.2

- Fixed the Hyperopt bookkeeping and warm starts, with regression tests
  for LHS validation and Hyperband bookkeeping.

## 0.2.1

- Added NEWS.md and a helpful error stub for the removed `GPSampler`.

## 0.2.0

### Breaking changes

- Removed `GPSampler`: Hyperopt removed `GPSampler` upstream in 0.5+, so the
  wrapper could not be kept. Use `RandomSampler`, `LHSampler` or `CLHSampler`
  instead.
- Hyperopt compatibility raised from 0.4 to 0.6. Hyperband objectives now
  follow the Hyperopt 0.6 protocol (a `state` keyword, real-valued budgets,
  and a NamedTuple result from `Hyperopt.hyperband`).
- Requires Julia 1.10+.
