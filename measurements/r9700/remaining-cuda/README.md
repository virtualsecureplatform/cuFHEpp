# Remaining CUDA optimization experiments

This directory contains the second R9700 pass. Each fingerprint directory stores
an immutable source manifest, independent feature builds, validation logs, and
fresh-key timing comparisons. `bench_r9700_remaining.py` generates these records.

The implementation and performance evaluation are in progress. Existing R9700
presets have not been promoted on the basis of these experiments.

The initial combined implementation passed independent exact arithmetic,
prepared key switching, scalar and paired external products, public/raw binary
gates, MUX/NMux, batch alias/padding checks, cooperative comparisons, and graph
replay. The focused checks additionally record the actual scalar-64,
register-31, and prepared-key-switch policies, avoiding a pass for a disabled
optimization. Final-source per-feature validation and performance qualification
are recorded separately in each configuration's receipts.

`capabilities/` contains the gfx1201 compiler probe, toolchain identity, and
selected native arithmetic disassembly. A probe value of -1 means the compiler
does not expose that builtin for the selected target. These probes do not claim
to test other AMD architectures.

The early `63dd14de3ac61cd3` build snapshot is explicitly superseded and contains
no qualified timing results. The corrected `2d79cdeec061447d` snapshot includes
an inverse-workspace reuse regression test: delaying upper-half peer readers
passes in three corrected runs and fails in three controls with the reuse
barrier removed. The control patch, compilation commands, executable hashes, and
all six logs are retained in `inverse-reuse-regression/`.

Source snapshots include the base commit, file hashes, source patch, and runner
copy. Binary hashes tie validation receipts to the executable files used for
measurements. Profile locks protect shared controls; build jobs share the global
lock, while GPU jobs take it exclusively to exclude compilation from timings.

The current snapshot's `resource_inventory.py` extracts unique kernel resource
shapes from completed exploratory measurements into `resource-inventory.json`.
Each shape retains an example log and reports registers, static/dynamic LDS,
private bytes, block size, and HIP's actual active-block estimate. These observed
launch attributes are separate from the requested minimum waves per EU.
`qualification_audit.py` checks full original-backend comparison coverage and
identifies threshold-adjacent results that need 30-repetition measurements;
partial exploratory coverage cannot qualify a production preset.

The original-backend audit also reports `unresolved_regressions` when 30-repetition
key runs still straddle the 2% regression threshold. Such a candidate remains
experimental. Completing a tuning sweep does not by itself qualify a preset;
final review must check full-grid coverage, repeatable intended gains, confirmed
regressions, and unresolved results together.

`00b58898b54a09cb` is an isolated, unadopted experiment that split the MUX
consumer schedule from binary-gate multi-output key switching. Its baseline and
both alternatives passed exhaustive correctness tests, but both alternatives
failed their 20-point, three-key, 30-repetition performance screens. The source
patch and complete records are retained for review; this extra selector is not
part of the main implementation. Its additional wave-selector test was cancelled
before GPU execution after that decision. The main snapshot's independent-wave
implementation and validation are unchanged. See that directory's
`consumer-schedule-critical30-summary.json` and `isolated-policy.json`.

The isolated directory also contains `runner-report-fix.patch`, an independent
fix for the runner's timing-record scan. Its regression fixture reproduces the
accidental inclusion of qualification arrays and verifies the scoped collector.
That fix is pending application after the main source freeze ends.
For the main worktree, `2d79cdeec061447d/runner-final-fixes.patch` is the preferred
combined patch: it also keeps generic FFT wave-budget variants on their baseline
prepared-key settings. Its regression checks pass without changing active source.
Apply the combined patch once, rather than applying both report patches.
