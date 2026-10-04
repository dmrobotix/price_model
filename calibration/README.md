# Calibration grid driver

Runs the two calibration grids that fix the elasticity `S` and the electricity
cost `C_elec` of the model, one grid point per process, in parallel, and
resumably.

## What it runs

The grids come from two existing scripts in `notebooks/`. Their grid loops now
sit under `if __name__ == "__main__":`. Importing them runs the data setup
only. Running them directly behaves as before.

| Grid | Script | Varied | Fixed (from `config.py` unless overridden) | RMSLE window (block heights) |
|---|---|---|---|---|
| `pre2018` | `sensitivity_analysis_pre2018_blocks.py` | `S_0`, `C_elec_0` | `S`, `C_elec` | [68633, 501995) |
| `modern` | `sensitivity_analysis_2018-2024_rmsle_blocks.py` | `S`, `C_elec` | `S_0`, `C_elec_0` | [501995, 877280) |

The grid is `S` in {0.01, ..., 0.10} (S = 0 is excluded unless `--include-s0`)
times `C` in {10, 20, ..., 150}: 150 points. The values come from `np.linspace`
exactly as in the scripts.

`run_grid_point.py` evaluates one point. It changes directory to `notebooks/`
(the model's data paths are relative to it), imports the grid script by file
path, and calls the script's own `objective_function` with the script's own
`initial_state`, `params`, `num_steps` and `H_hist_series`.

`run_grid.py` is the orchestrator. It starts one `run_grid_point.py` process per
point, at most `--jobs` at a time.

## The two phases, and why phase 2 uses the phase-1 optimum

The early era (before block 501995) and the modern era are fitted in sequence.
Phase 2 simulates from block 0, so its modern-era hashrate depends on the
early-era parameters `S_0` and `C_elec_0` that were used to get there. The
modern grid therefore needs a value for them.

- **Phase 1** evaluates every `pre2018` point, with `S` and `C_elec` at the
  config values. It writes `grid_pre2018.csv` and `phase1_optimum.json`, the
  minimum-RMSLE point. Crashed points are ignored when picking it.
- **Phase 2** evaluates every `modern` point with `S_0` and `C_elec_0` set to the
  phase-1 optimum. It writes `grid_2018_2024.csv` and `phase2_optimum.json`.

Without the override, the modern grid would use whatever `S_0` and `C_ELEC_0` are
in `config.py` at the time, which need not be the optimum of a pre-2018 grid
computed with the current code.

Each modern result row records `fixed_S_0` and `fixed_C_elec_0`. On resume, the
orchestrator refuses to reuse modern points computed with a different
phase-1 optimum from the one in `phase1_optimum.json`.

## Usage

    /home/jynurso/miniconda3/envs/miner/bin/python calibration/run_grid.py \
        --out-dir <OUT_DIR> --jobs 4 \
        --python /home/jynurso/miniconda3/envs/miner/bin/python

Options: `--phase {1,2,both}` (default both), `--include-s0`, `--jobs N`,
`--python PATH`, `--retry-crashed`.

Launch it with `systemd-run --user` (or inside tmux or screen), not with `nohup`.
The orchestrator handles SIGHUP by stopping every child, so a `nohup` launch
from an ssh shell stops the whole grid when the session closes. Use a fresh
`--out-dir` for each new grid, because the phase-2 gate checks the crashed and
missing counts in `phase1_optimum.json` but not the size of the grid.

Each child process peaks at about 2.2 GB of memory (measured: 1.2 GB after data
loading, 1.6 GB at step 520,000, extrapolated to about 1.9 to 2.2 GB at step
888,300), so `--jobs` multiplied by 2.2 GB must fit in free memory. Data
loading takes 65 to 105 s per child, and the model run takes tens of minutes.
The `peak_rss_mb` and `elapsed_s` columns in the point CSVs report the actual
values for each point. Children run with `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1, so the parallelism comes
from `--jobs` alone.

## Retry and the phase gate

- **In-invocation retry.** After a phase's first pass, every point attempted in
  this invocation that has no result (killed child) or a crashed result
  (RMSLE = 1e9) is run once more through the same `--jobs` pool. The progress
  log shows `RETRY` lines.
- **Phase gate.** If a phase-1 point is still missing or crashed after that
  retry, phase 2 does not start. `phase1_optimum.json` is still written, with
  `n_points_crashed` and `n_points_missing`. An `ERROR` line names the points
  and the exit status is non-zero. `--phase 2` on its own also refuses to start
  from a `phase1_optimum.json` that records crashed or missing points. At the
  end of phase 2 the same completeness report is written.
- **Edge of the grid.** Each optimum JSON has `on_grid_edge`. It is true when the
  optimum has the smallest or largest S, or the smallest or largest C, of the
  grid that was run. A `WARNING` line is logged, because the true optimum may lie
  outside the grid.

## Resuming

Run the same command again. A point is skipped when its result CSV exists in
`points/`. A point whose process was killed (no CSV) is run again. A point that
returned the crash sentinel in an earlier invocation has a CSV, so it is skipped
unless `--retry-crashed` is given; it then counts as crashed for the phase gate.
`--phase 2` alone reads `phase1_optimum.json` from the same `--out-dir`.

SIGINT, SIGTERM or SIGHUP to the orchestrator terminates its children and
exits. Result CSVs are written atomically (temp file, then `os.replace`), so a
point is either complete or absent.

After an abnormal stop of the orchestrator itself (for example SIGKILL), check
for orphaned children before resuming, because they keep running and would
write the same points:

    pgrep -f run_grid_point.py

The exit status is 1 if any point is crashed or has no result, 2 for fatal
errors, and 0 otherwise. A crash does not stop the other points of the phase.

## Output layout

    <out-dir>/
      progress.log            timestamped start/finish/elapsed/RMSLE per point, done/remaining counts
      logs/<grid>_S0.05_C60.log   stdout and stderr of one child
      points/<grid>_S0.05_C60.csv one-row result per point
      grid_pre2018.csv        merged phase-1 results
      phase1_optimum.json
      grid_2018_2024.csv      merged phase-2 results
      phase2_optimum.json

Point CSV columns: `grid`, `S_0`/`C_elec_0` (pre2018) or `S`/`C_elec` (modern),
`RMSLE`, `fixed_S_0`, `fixed_C_elec_0` (modern only), `num_steps`, `elapsed_s`
(objective only), `setup_s` (data loading in the child), `peak_rss_mb`,
`git_head`, `git_dirty`.

### Provenance

`git_head` and `git_dirty` come from git when `calibration/..` is the top level of
a git work tree. `git_dirty` is then true when tracked files differ from HEAD;
untracked files are ignored. A directory nested inside some other repository does
not count, so the parent repository's HEAD is never recorded.

For code exported with `git archive` (no `.git`), create a file `PROVENANCE` in
the code root with two lines, then `git_head` is the commit and `git_dirty` is
`archive`:

    commit=<full sha>
    source=git-archive

With neither git nor `PROVENANCE`, both columns are `unknown`.

## Notes

- Existing results in `data/results/` and `notebooks/` are never written.
  Use a fresh `--out-dir`.
- Every child starts from a fresh process. The serial scripts reuse one process
  for all 150 runs and reset only the caches that `objective_function` clears,
  so any other state carried between runs there is absent here.
- `--num-steps` (child) and `--max-points`, `--child-num-steps` (orchestrator)
  exist for testing. A run with fewer than 888,300 steps does not cover the
  RMSLE window: the pre2018 window needs steps beyond 68,633 and the modern
  window beyond 501,995; otherwise `objective_function` returns 1e9.
