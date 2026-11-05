# NEWS

## 0.2.0

### Breaking changes

- Removed `GPSampler`: Hyperopt removed `GPSampler` upstream in 0.5+, so the
  wrapper could not be kept. Use `RandomSampler`, `LHSampler` or `CLHSampler`
  instead.
- Hyperopt compatibility raised from 0.4 to 0.6. Hyperband objectives now
  follow the Hyperopt 0.6 protocol (a `state` keyword, real-valued budgets,
  and a NamedTuple result from `Hyperopt.hyperband`).
- Requires Julia 1.10+.
