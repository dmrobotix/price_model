# Scenario runs

This directory runs the PRICE model's ten scenario runs without editing
`main.py` or `config.py` by hand. The ten runs are the out-of-sample test, the
Figure 4 hindcast, and the eight forecast scenarios of Figures 6 to 9. Each run
is one process. Several runs can execute in parallel. An interrupted set of runs
can be resumed.

| File | Purpose |
|---|---|
| `runs.json` | The run definitions: settings and output name of each run. |
| `run_scenarios.py` | The orchestrator. It starts one `run_one.py` process per run, at most `--jobs` at a time. |
| `run_one.py` | Runs `main.py` once, checks the values in effect, and writes the completion marker. |
| `verify_reproduction.py` | Compares a run's CSV with a saved original, row by row. |
| `prepare_april_reproduction.py` | Exports the April 2026 code and patches it to re-run one original scenario. |
| `analyze_results.py` | Computes the paper's statistics and draws Figures 4 to 9 from the ten run CSVs of one set. |

## The ten runs

| Run | `num_steps` | `historical_cutoff` | Efficiency | Forecast model | Target price (2032-01-01) | Output name |
|---|---|---|---|---|---|---|
| `testing` | 921,683 | 2025-01-01 00:00:00 | frontier | powerlaw | 1,000,000 | `simulation_results_testing_2026-10` |
| `hindcasting` | 877,259 | 2010-07-17 00:00:00 | frontier | powerlaw | 1,000,000 | `simulation_results_hindcasting_2026-10` |
| `frozen_price_fixed` | 1,260,000 | 2025-11-08 17:58:32 | frozen | fixed | none | `simulation_results_efficiency_frozen_price_fixed_2026-10` |
| `frozen_price_200k` | 1,260,000 | 2025-11-08 17:58:32 | frozen | powerlaw | 200,000 | `..._frozen_price_200k_2026-10` |
| `frozen_price_500k` | 1,260,000 | 2025-11-08 17:58:32 | frozen | powerlaw | 500,000 | `..._frozen_price_500k_2026-10` |
| `frozen_price_1M` | 1,260,000 | 2025-11-08 17:58:32 | frozen | powerlaw | 1,000,000 | `..._frozen_price_1M_2026-10` |
| `frontier_price_fixed` | 1,260,000 | 2025-11-08 17:58:32 | frontier | fixed | none | `..._frontier_price_fixed_2026-10` |
| `frontier_price_200k` | 1,260,000 | 2025-11-08 17:58:32 | frontier | powerlaw | 200,000 | `..._frontier_price_200k_2026-10` |
| `frontier_price_500k` | 1,260,000 | 2025-11-08 17:58:32 | frontier | powerlaw | 500,000 | `..._frontier_price_500k_2026-10` |
| `frontier_price_1M` | 1,260,000 | 2025-11-08 17:58:32 | frontier | powerlaw | 1,000,000 | `..._frontier_price_1M_2026-10` |

Every run reads `combined_block_data.csv` and `market_price_min.csv` (see
"Input files" below).

`runs.json` also holds an eleventh run, `repro_april_frozen_price_fixed`. It has
`"default": false`, so it runs only when named with `--runs`. It is described
under "Reproduction check" below.

### Boundaries are UTC instants (2026-10-05)

Every period boundary is a UTC instant, and its boundary block is the first block
whose `Block_Time_Seconds` is later than the instant. That block is the last block
of the earlier period (`modules/boundaries.py`). This is how `simulation.py`'s
historical/forecast switch already behaved.

| Instant (UTC) | Boundary block | Next block starts |
|---|---|---|
| 2010-07-17 | 68,607 | the economic model in `hindcasting` (68,608) |
| 2018-01-01 | 501,961 | the modern era, S and C_elec (501,962) |
| 2025-01-01 | 877,259 | the out-of-sample test (877,260) |
| 2025-11-01 | 921,683 | (end of `testing`) |
| 2025-11-08 17:58:32 | 922,756 | the forecast in the eight scenarios (922,757) |

`hindcasting` now ends at block 877,259 (it ended at 877,280, the first block
stamped after Eastern midnight). A run with `end_boundary_utc` in `runs.json`
(`testing`, `hindcasting`) must have `num_steps` equal to that instant's boundary
block: `run_one.py` computes it from the run's block data and stops with exit
status 3 before the simulation if it differs. `--child-num-steps` test runs skip
the check.

### Where the settings come from

The text below describes how the October 2026 settings were read off the
original outputs. `hindcasting`'s `num_steps` has since changed (see above).

Each setting was read off the saved original output in `data/results/`, not
taken from comments in `main.py`. The originals were read in chunks with only
the needed columns.

- **`num_steps`.** The last `block_height` of each original: 921,683 for
  `testing`, 877,280 for `hindcasting`, 1,260,000 for each of the eight scenarios.
- **`historical_cutoff`.** `simulation.py` switches block `h` to forecast mode
  when the timestamp of block `h-1` is later than the cutoff. Forecast-mode rows
  are the rows with a non-empty `mean_bt`. So the cutoff lies at or after the
  latest timestamp of blocks `0 .. h-2`, and strictly before the timestamp of
  block `h-1`, where `h` is the first forecast-mode block. Every cutoff in that
  interval gives the same output.
  - `testing`: first forecast block 877,260; interval
    [2024-12-31 23:56:16, 2025-01-01 00:21:15). 2025-01-01 00:00:00 lies in it.
  - `hindcasting`: first forecast block 68,608; interval
    [2010-07-16 23:55:58, 2010-07-17 00:11:43). 2010-07-17 00:00:00 lies in it.
  - All eight scenarios: first forecast block 922,757; interval
    [2025-11-08 17:58:32, 2025-11-08 18:04:29). 2025-11-08 17:58:32, the last
    transaction-fee time, is the lower end and lies in it.
- **Forecast model and target price.** The forecaster prices a block when the
  timestamp of the block before it is later than `last_hist_time`
  (2025-11-08 17:58:32). In the eight scenarios that holds for 337,244 rows.
  - The two `fixed` runs have one `P_USD` value in all of those rows: 91,821.06.
    That is the last price in `market_price_min.csv`. `build_price_forecaster`
    returns that constant for `FORECAST_MODEL = "fixed"` with
    `FORECAST_TARGET_PRICE = None`.
  - In the six `powerlaw` runs, `P_USD` in the first forecaster-priced row equals,
    to the last bit, the two-point power law through (2025-11-27 18:59:00,
    91,821.06) and (2032-01-01, target price). The values are 91,106.61173732682
    (200k), 90,272.8023818642 (500k) and 89,647.12348336326 (1M). The row nearest
    2032-01-01 is within 3 of the target price in each run.
  - In every forecaster-priced row, `TX_fee` equals 4% of `R_block`, which is the
    forecast fee rule.
- **Efficiency scenario.** For each price, the frozen and frontier originals
  have identical `efficiency` up to block 939,083 or later (the first difference
  is at block 939,084 for fixed and 200k, 939,087 for 500k, 939,101 for 1M;
  block 939,084 is stamped 2026-03-01 00:11:41). The frozen runs end at a single
  constant efficiency over their last 100,000 blocks. The frontier runs keep
  falling, to about 2.8 W/TH. This matches the frontier machine entering only
  after the last machine's deployment date.
- **Settings without an effect in `testing` and `hindcasting`.** Neither original
  has a forecaster-priced row: `testing` ends at 2025-10-31 23:33:03 and
  `hindcasting` at 2024-12-12 05:48:22, both before `last_hist_time`. The frontier
  machine also needs a date past `last_hist_time`. So the forecast model, target
  price and efficiency scenario of those two originals cannot be read from their
  output. `runs.json` sets them to today's `config.py` values. They change the
  re-run only if its simulated clock passes 2025-11-08 17:58:32 within its
  `num_steps`. `run_one.py` reports this as `forecaster_priced_rows` for every run.

### Input files: the re-runs use the originals' files

Every run in `runs.json` sets `block_pace_data = ../data/combined_block_data.csv`
and `price_data = ../data/market_price_min.csv`. This was the authors' decision on
2026-10-02: the re-runs use the input files the paper's runs used. Since
2026-10-06 the `config.py` defaults are the same pair, so a run with no `PRICE_*`
variables reads these files too. Until then the defaults (and the April tag's)
were `market_price_min_latest.csv` and `combined_block_data_latest.csv`, and the
calibration grids read `combined_block_data_latest.csv` (see `config.py`).

The eight scenario originals were produced from `market_price_min.csv` and
`combined_block_data.csv`. The evidence:

- The `fixed` forecast price, 91,821.06, is the last row of `market_price_min.csv`
  (2025-11-27 18:59:00). The last row of `market_price_min_latest.csv` is
  67,990.95 (2026-02-12 12:28:00). The power-law values above also match only the
  older file.
- The originals' `H_hist` column ends at block 925,641, the last block of
  `combined_block_data.csv`. The latest file runs to block 936,248.

On the blocks and minutes the two pairs share, they agree exactly: no difference
in `Block_Time_Seconds` or `Bits` over blocks 0 to 925,641, and no price
difference over the common minutes. The choice therefore changes only:

1. the forecast price in all eight scenarios (the `fixed` level, and the starting
   point of the two-point power law), and
2. the `Hist_Timestamp`, `H_hist`, `T_hist_seconds`, `P_hist` and `E_hist`
   columns for blocks 925,642 to 936,248.

The historical part of every run, and so the calibration, is unaffected.

For `testing` and `hindcasting` the output cannot show which pair was used.
Neither run has a forecaster-priced row. Both end before block 925,641
(`H_hist` is present up to the last block, 921,683 and 877,280). `H_hist` at a
block uses only the 2,016 blocks up to it, and the two block files agree on
those. On nyxnyx both originals have a modification time of 5 January 2026,
before the `_latest` files' 12 February 2026, which is consistent with the older
pair (a modification time can change on copying, so this is weak evidence). `runs.json`
gives them the older pair for consistency with the eight scenarios.

Every run records the input files' paths, sizes and SHA-256 in its
`settings.json` and `DONE.json`.

## How the settings reach the model

`config.py` and `main.py` read optional environment variables. With none set,
every value is exactly the hardcoded one, and `main.py` runs as before.

| Variable | Read in | Overrides | Accepted values |
|---|---|---|---|
| `PRICE_EFFICIENCY_SCENARIO` | `config.py` | `EFFICIENCY_SCENARIO` | `frozen`, `frontier` |
| `PRICE_FORECAST_MODEL` | `config.py` | `FORECAST_MODEL` | `powerlaw`, `fixed`, `constant`, `linear`, `logistic` |
| `PRICE_FORECAST_TARGET_DATE` | `config.py` | `FORECAST_TARGET_DATE` | ISO date or date-time without UTC offset |
| `PRICE_FORECAST_TARGET_PRICE` | `config.py` | `FORECAST_TARGET_PRICE` | `none` (any case) or a positive finite number |
| `PRICE_BLOCK_PACE_DATA` | `config.py` | `BLOCK_PACE_DATA` | `<name>.csv` or `../data/<name>.csv`; no other directory |
| `PRICE_PRICE_DATA` | `config.py` | `PRICE_DATA` | `<name>.csv` or `../data/<name>.csv`; no other directory |
| `PRICE_NUM_STEPS` | `main.py` | `num_steps` | positive integer |
| `PRICE_HISTORICAL_CUTOFF` | `main.py` | `params['historical_cutoff']` | ISO date or date-time without UTC offset |
| `PRICE_OUTPUT_NAME` | `main.py` | `file_name` | `[A-Za-z0-9][A-Za-z0-9._-]*`, at most 200 characters, not ending in `.csv` |

A malformed value raises `ValueError` when `config.py` is imported, which is
before any data is loaded. So does any other variable whose name starts with
`PRICE_`. That catches misspelt names. `config.py` sets
`EFFICIENCY_SCENARIO` before `modules/efficiency_cbeci.py` imports it, so the
override reaches that module as well. `TX_BLOCK_DATA` and `MACHINE_DATA_FILE`
have no override. No module, no line of `main.py` and no default in `config.py`
names a `_latest` file.

When any of the three `main.py` variables is set, `main.py` prints one line
`[run overrides] num_steps=... historical_cutoff=... file_name=...`.

The orchestrator always sets all nine variables for a child, from `runs.json`,
and removes any other `PRICE_*` variable from the child's environment.

## Running on archylnx

### 1. Export the code

`git archive` does not need a work tree on archylnx. Export only the code. The
repository also tracks `data/`, about 5.9 GB.

    SHA=$(git -C /mnt/storage/research-data/miner-economics-simulator rev-parse HEAD)
    OUT=code-${SHA:0:7}
    mkdir $OUT
    git -C /mnt/storage/research-data/miner-economics-simulator archive $SHA \
        main.py config.py modules scenarios calibration/run_grid_point.py | tar -x -C $OUT
    printf 'commit=%s\nsource=git-archive\n' $SHA > $OUT/PROVENANCE
    rsync -a $OUT archylnx:price-scenarios/

The exported commit must contain this directory and the `config.py`/`main.py`
overrides, so export after they are committed. `run_one.py` and
`run_scenarios.py` load `git_state()` from `calibration/run_grid_point.py` for
the provenance record.

### 2. Input files

`--data-dir` must hold the four files the runs read:
`combined_block_data.csv`, `market_price_min.csv`, `txfee_data.csv` and
`cbeci_machines_090325.csv`. The reproduction run reads the same four. For each
run the orchestrator imports `config.py` of the code root with that run's
input-file settings, and takes the four names from it. If any file of any
selected run is missing, it lists them and stops before starting anything.

### 3. Launch

Launch with `systemd-run --user`, or inside tmux or screen. Do not use `nohup`.
The orchestrator treats SIGHUP like SIGTERM and stops every child, so a `nohup`
launch from an ssh shell stops the runs when the session closes.

    systemd-run --user --unit=price-scenarios-2026-10 --collect \
        -p WorkingDirectory=$HOME/price-scenarios \
        systemd-inhibit --what=sleep:idle --why="PRICE scenario runs" \
        $HOME/micromamba/envs/price/bin/python \
        $HOME/price-scenarios/code-XXXXXXX/scenarios/run_scenarios.py \
            --out-dir $HOME/price-scenarios/out-2026-10 \
            --data-dir $HOME/price-grid/data \
            --jobs 2 \
            --python $HOME/micromamba/envs/price/bin/python

This command was tested on nyxnyx only. There `systemd-run` started the
orchestrator, but `systemd-inhibit` exited with "Failed to inhibit: Access
denied", which ends the unit at once. The calibration grid ran under
`systemd-inhibit` on archylnx. If it is refused there, drop the
`systemd-inhibit ...` line and keep the machine from sleeping by other means.

Follow it with `tail -f $HOME/price-scenarios/out-2026-10/progress.log` or
`journalctl --user -u price-scenarios-2026-10 -f`. Stop it with
`systemctl --user stop price-scenarios-2026-10`. A `--user` unit stops when the
user's last session ends unless lingering is enabled for that user.

Options:

- `--runs default|all|<name>,<name>` selects runs. `default` is every run with
  `"default": true`, which is the ten runs.
- `--jobs N` is the number of runs at a time. See "Memory and time".
- `--code-root DIR` is the tree with `main.py` and `config.py`. The default is the
  tree that holds this directory.
- `--data-dir DIR` defaults to `<code-root>/data`.
- `--rss-interval S` sets how often each child prints its memory use (default 120 s).

`--out-dir` may not lie inside `--data-dir`, `<code-root>/data` or this
repository's `data/`. The orchestrator stops if it does.

## Memory and time

Peak memory per run was measured on nyxnyx with short runs of HEAD code:

| Run | Steps | Peak RSS |
|---|---|---|
| `testing` | 30,000 | 1,482 MB |
| `hindcasting` | 30,000 | 1,466 MB |
| `hindcasting` | 150,000 | 1,649 MB |

Data loading reaches about 1.1 to 1.3 GB. The peak comes after the simulation,
when `main.py` builds the result table, merges the historical series and writes
the CSV. Between 30,000 and 150,000 steps the peak grows by 1.4 to 1.5 MB per
1,000 steps. A straight-line extrapolation gives:

| Steps | Runs | Estimated peak |
|---|---|---|
| 877,280 | `hindcasting` | 2.7 to 2.8 GB |
| 921,683 | `testing` | 2.7 to 2.8 GB |
| 1,260,000 | eight scenarios | 3.2 to 3.4 GB |

These are extrapolations from two run lengths, not measurements of full runs.
The calibration grid measured about 2.2 GB at 888,300 steps on archylnx, but it
does not build `main.py`'s result table. Budget 4 GB per run. On archylnx
(11.5 GB) that means `--jobs 2`. With `--jobs 3`, three runs reaching their peaks
at the same time would need about 10 GB, which risks a child being killed for memory.
Run only while logged in on archylnx (`Linger=no`): `systemd-run --user` units stop at
logout. `loginctl enable-linger` removes that limit. Each child prints its memory every
`--rss-interval` seconds, and `DONE.json` records the measured `peak_rss_mb`.

On nyxnyx, 30,000 steps took about 340 s per run with two runs on two cores, of
which 138 s was the simulation loop. archylnx ran calibration points about 7
times faster than nyxnyx.

## Output layout

    <out-dir>/
      progress.log                 START / FINISH / FAILED / RETRY lines with elapsed time and done/remaining counts
      summary.json                 final status of every selected run
      runs/<name>/
        settings.json              requested settings, PRICE_* values, input manifest (path, size, SHA-256), commit
        stdout.log                 stdout and stderr of every attempt, appended; includes main.py's CBECI comparison
        DONE.json                  completion marker (below); present only for a complete run
        work/                      the child's working directory (stays empty)
        data/
          <input files>            symlinks to the files in --data-dir
          <output_name>.csv        the result
          hashrate_comparison.png  written by main.py
          logs/model_debug.log     written by main.py

Each child prints, before the simulation starts: host, code root, commit, the
`PRICE_*` values it received, the requested settings, and the `config.py` values
in effect (`S`, `C_ELEC`, `S_0`, `C_ELEC_0`, `T_STAR`, `EFFICIENCY_SCENARIO`,
`FORECAST_MODEL`, `FORECAST_TARGET_DATE`, `FORECAST_TARGET_PRICE`,
`CALIBRATION_MODE` and the four input paths). If a `config.py` value differs from
the requested setting, the child stops with exit status 3 before loading data.

After `main.py` returns, the child compares `main.py`'s own `num_steps`,
`historical_cutoff` and `file_name` with the request, reads the CSV's last line,
and requires the last block to equal `num_steps`. Only then does it write
`DONE.json`, atomically. `DONE.json` holds the settings, the input manifest, the
values in effect, the commit, CSV path and size, row count, last timestamp, the
first forecast-mode block, `forecaster_priced_rows`, the last row of the price
file, elapsed time and peak memory.

## Resuming, failures and signals

Run the same command again. A run is skipped when its `DONE.json` exists, its
settings, inputs (path, size, SHA-256) and commit equal the current ones, and the
CSV size equals the recorded size. A `DONE.json` that disagrees on any of these
stops the orchestrator with an error rather than reusing the run or running over
it. Use a fresh `--out-dir`, or delete that run's directory.

A run without `DONE.json` is run again. A partial CSV left by an interrupted
attempt is deleted first. Within one invocation, each run that fails is retried
once, after the first pass over all runs. The exit status is 0 when every
selected run is complete, 1 when any failed, and 2 for a fatal error.

SIGINT, SIGTERM and SIGHUP to the orchestrator terminate its children (SIGKILL
after 10 s) and exit. Completed runs stay on disk. Children run in their own
session, so a Ctrl-C in the terminal reaches only the orchestrator.

Each child takes a lock on its run directory, and the orchestrator takes a lock
on `--out-dir`. If the orchestrator itself is killed with SIGKILL, its children
keep running. A new orchestrator can start, but a child it launches for a run
that an orphan still holds exits with status 5. Check for orphans with
`pgrep -f scenarios/run_one.py`.

Children run with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS`
set to 1, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1` and `MPLBACKEND=Agg`.

## Reproduction check

Before the new runs are trusted, one original scenario is re-run with the April
2026 code (tag `paper3-april-2026`) and compared with the saved original. The
scenario is `efficiency_frozen_price_fixed`. The April code has no `PRICE_*`
overrides, so its hardcoded lines are patched in an export.

1. On nyxnyx (needs git):

        python scenarios/prepare_april_reproduction.py --export-dir <new dir>

   This exports `main.py`, `config.py` and `modules/` of the tag, writes
   `PROVENANCE`, and changes exactly eight lines. Each line must match its
   expected April text exactly once, or nothing is patched. The diff is printed
   and saved as `PATCH.diff`:
   - `main.py`: `num_steps = 1_260_000`, `historical_cutoff` 2025-11-08 17:58:32,
     `file_name = "simulation_results_efficiency_frozen_price_fixed_repro-april"`;
   - `config.py`: `FORECAST_MODEL = "fixed"`, `FORECAST_TARGET_PRICE = None`,
     `EFFICIENCY_SCENARIO = "frozen"`, and the originals' input files
     `BLOCK_PACE_DATA = '../data/combined_block_data.csv'` and
     `PRICE_DATA = '../data/market_price_min.csv'`.

   `FORECAST_TARGET_DATE`, `TX_BLOCK_DATA` and `MACHINE_DATA_FILE` are checked
   and left unchanged.

2. Copy the export to archylnx. Copy `combined_block_data.csv` and
   `market_price_min.csv` into the data directory used for this run, next to
   `txfee_data.csv` and `cbeci_machines_090325.csv`. Copy the original
   `data/results/simulation_results_efficiency_frozen_price_fixed.csv`.

3. Run it with this repository's orchestrator, pointing `--code-root` at the
   patched export:

        python <code>/scenarios/run_scenarios.py --out-dir <out-repro> \
            --code-root <patched export> --data-dir <data dir> \
            --runs repro_april_frozen_price_fixed \
            --python $HOME/micromamba/envs/price/bin/python

   For code without the overrides, `run_one.py` reads the hardcoded
   `num_steps`, `historical_cutoff` and `file_name` from `main.py` before
   starting and stops if they differ from the request. The run's
   `block_pace_data` and `price_data` settings require the patched `config.py`
   to name the two older input files.

4. Compare:

        python <code>/scenarios/verify_reproduction.py \
            <out-repro>/runs/repro_april_frozen_price_fixed/data/simulation_results_efficiency_frozen_price_fixed_repro-april.csv \
            <originals>/simulation_results_efficiency_frozen_price_fixed.csv

   The script hashes both files, then compares `block_height`, `Timestamp`,
   `H_sim`, `efficiency`, `P_USD`, `E_sim` and `R_pe` row by row as text, and
   reports maximum absolute and relative differences for any column that
   differs. `--all-columns` compares every shared column. Exit status 0 means
   every compared value is identical. `--head N` compares only the first N rows.

## Testing options

`run_scenarios.py --child-num-steps N` runs every selected run with
`num_steps = N`. The settings record N, so a test directory is never mistaken
for full runs. `prepare_april_reproduction.py --num-steps N` patches a short
April run whose output name ends in `_steps<N>`. Both options are hidden from
`--help`.

Test runs need N of at least a few thousand steps; 30,000 is known to work. At
1,000 steps `main.py` itself fails in `plot_annual_energy`: `H_hist` is empty
before block 2,015, so the annual energy series it plots has no rows.

## Statistics and Figures 4 to 9: `analyze_results.py`

`analyze_results.py` replaces the cells of `data/results/scenario_results.ipynb`
that produce the paper's statistics and Figures 4 to 9. It reads the ten run
CSVs of one set and writes everything into a new directory. It refuses to write
into a directory that exists. It writes into a temporary sibling directory
(`.<name>.partial-*`) and renames it to `--out-dir` only when the run has
finished. A failed run therefore leaves no `--out-dir`, and it can be rerun with
the same name. Before the rename, the directory is given the mode a plain
`mkdir` would give it (`0777` minus the umask). `mkdtemp` creates it with mode
`0700`, and that mode would also set the ACL mask to `---`, which would block
users such as `www-data` that a default ACL on the parent grants access. The
parent of `--out-dir` is created if it does not exist.

    python scenarios/analyze_results.py --set original --out-dir data/results/analysis_2026-10/original_v2
    python scenarios/analyze_results.py --set new      --out-dir data/results/analysis_2026-10/new_v2
    python scenarios/analyze_results.py --compare data/results/analysis_2026-10/original_v2 \
        data/results/analysis_2026-10/new_v2 --out-dir data/results/analysis_2026-10/compare_v2
    # the notebook's definitions:
    python scenarios/analyze_results.py --set original --clock simulated --hindcast-window notebook \
        --out-dir data/results/analysis_2026-10/original_simclock_nbwindow

The two sets:

- `original` reads the CSVs in `data/results/` named in the `original` field of
  `runs.json`. These are the runs behind the paper's numbers. The electricity
  cost is `C_ELEC` in `config.py` at git tag `paper3-april-2026` ($50/MWh).
- `new` reads `<runs-dir>/<name>/data/<output_name>.csv`. The default `--runs-dir` is
  `data/results/scenarios_2026-10-rerun/runs`, the runs behind the paper's numbers.
  Until 2026-10-06 it was `data/results/scenarios_2026-10/runs`, the October 2026
  runs made before the clock and UTC fixes; pass `--runs-dir` to analyse those.
  Each run must have its `DONE.json`, and the CSV size must equal the size
  recorded there. The electricity cost is `config_in_effect.C_ELEC` from the
  `DONE.json` files, which must agree ($40/MWh for both the October 2026 runs and the re-run).

`--csv NAME=PATH` replaces one run's CSV. The size check against `DONE.json` is
skipped for a CSV given this way, because it is not that run's output.
`stats.json` records the check's result for every run
(`provenance.inputs.<name>.done_size_check`). `--c-elec` replaces the
electricity cost. `--no-figures` skips the figures.

The electricity cost C_elec is the break-even of R_pe. Every output that says
"break-even" means C_elec. The share of forecast blocks below $50/MWh is also
reported, under the name "below $50/MWh". $50/MWh was the break-even only for
the original runs.

### Two settings for comparisons with history

`--clock historical|simulated` (default `historical`) selects the time column on
which simulated and historical hashrate are compared. It applies to:

- the out-of-sample monthly Sim/Hist ratio and its window
  (2025-01-01 to 2025-10-31 23:59:59);
- the hindcast monthly Sim/Hist ratios and their two windows;
- the weekly and monthly means of the 2018-2024 hindcast metrics;
- Figures 4 and 5.

With `historical`, both series are binned and plotted by `Hist_Timestamp`, the
real time of each block. With `simulated`, they use `Timestamp`, the model's
clock, as the notebook did. The forecast scenario statistics and Figures 6 to 9
always use `Timestamp`, because they are outputs in model time.

`--hindcast-window calibration|notebook` (default `calibration`) selects the
block window of the 2018-2024 hindcast metrics (block-level, weekly and monthly).

| Value | Blocks | Hashrate units | Source |
|---|---|---|---|
| `calibration` | [501,962, 877,260) | TH/s (multiplied by 1e-12 before `log1p`) | the objective function of the 2018-2024 calibration grid, `notebooks/sensitivity_analysis_2018-2024_rmsle_blocks.py`; heights from the UTC boundary blocks, computed from the block data at run time |
| `calibration-eastern` | [501,995, 877,280) | TH/s | the window of the grids run before 2026-10-05 (Eastern midnight) |
| `notebook` | [505,227, 877,280) | H/s | notebook cells 14 to 16; 505,227 is where the original hindcast's simulated clock reached 2018-01-01 |

`--grid-optimum PATH` names the `phase2_optimum.json` whose RMSLE is checked
against the `new` set's hindcast (default: `data/results/grid_2026-10-rerun/`,
the grid whose optimum `config.py` holds; until 2026-10-06 it was the 2026-10-01
grid). The reference
applies under the window the grid recorded (`window_start_height`,
`window_end_height_excl`), or under `calibration-eastern` for a grid that
recorded none. `--compare` also refuses two outputs whose 2018-2024 height
windows differ.

The units change the RMSLE only in the ninth significant digit. They are set to the
grid's so that the calibration-window RMSLE reproduces the grid's value exactly.

`--set original --clock simulated --hindcast-window notebook` reproduces the
notebook. Each notebook reference value in `verification.csv` carries the
condition under which the notebook computed it. A reference whose condition does
not hold for the run is listed with `applies = False` and is left out of the
count.

Every CSV is read in chunks of 200,000 rows, with only the columns needed. Of
the scenario files, only the rows from 2025-11-01 onward are kept in memory.

### Which notebook cell made which figure

The images in the manuscript are byte-identical (md5) to PNGs the notebook
saved in `data/results/`.

| Figure | Cell | Notebook PNG | Run CSVs |
|---|---|---|---|
| 4 | 18 | `figure_hashrate_hindcasting_windows_with_monthly_ratio.png` | hindcasting |
| 5 | 20 | `figure_hashrate_testing_2025_with_monthly_ratio.png` | testing |
| 6 | 3 | `figure_efficiency_over_time_2026plus.png` | the eight scenarios |
| 7 | 23 | `figure1_hashrate_EHs__.png` | the eight scenarios |
| 8 | 1 (second figure) | `figure2_annual_energy_TWh.png` | the eight scenarios |
| 9 | 2 | `figure3_Rpe_line.png` | the eight scenarios |

### Statistics

| Output key | Definition | Notebook cell |
|---|---|---|
| `oos.*` | Log metrics of `H_sim` against `H_hist` over `block_height >= 877,260`: clip at 0, `log1p`, residual = `log1p(sim) - log1p(hist)`. `expm1_*` are the multiplicative readings. | 11 to 13 |
| `oos.monthly_ratio_*` | Monthly mean `H_sim` / monthly mean `H_hist`, 2025-01-01 to 2025-10-31 23:59:59, on the `--clock` column. | 20 |
| `hindcast.era1`, `hindcast.era2` | The same monthly ratio in the two windows of Figure 4, 2010-07-17 to 2017-12-31 and 2018-01-01 to 2024-12-31, on the `--clock` column, with the centred 3-month rolling mean. | 18 |
| `hindcast.metrics_2018_2024` | Log metrics over the `--hindcast-window` block window, at block level and on weekly (`W-SUN`) and monthly means on the `--clock` column. | 14, 16 |
| `annual_energy_TWh` | Sum of `E_sim` (Wh per block) by `Timestamp` year, divided by 1e12. 2025 and 2032 are partial years. | 1 |
| `efficiency_mean_by_year` | Mean of `efficiency` (W/TH, which equals J/TH) by `Timestamp` year. | CODE_AUDIT B1 |
| `peak_Rpe_forecast`, `share_Rpe_below_c_elec` (below break-even), `share_Rpe_below_50` (below $50/MWh) | Over forecast rows, which are the rows with a non-null `mean_bt`. "Below" is strict. | (audit, Part G) |
| `halving_timestamp` | `Timestamp` of block 1,050,000. | 24 |
| `peak_H_EHs_2026_2031` | Maximum `H_sim` with `Timestamp` in [2026-01-01, 2031-12-31 00:00:00]. | 21 |
| `linearity` | Linear and quadratic R² of monthly mean `H_sim` on time, split at the halving timestamp. Each period runs from its own first row (`t` re-zeroed), so the halving month is partial in both periods. | 22 |
| `checks.*` | Subsidy halving at block 1,050,000; frontier above frozen at each price; hashrate increasing in price. Each is checked on the peak, on annual means, and on monthly means of the forecast rows by calendar month (the first, mixed month and the last, partial month are left out). Months are compared rather than block heights, because two runs reach the same block height at different times. | (audit, Part F) |

Cell 22 used a `FORECAST_START` from a kernel state the notebook does not show.
Every start from 2025-11-08 00:00 to 2025-11-09 00:00 reproduces its printed
output, and the script uses 2025-11-08 17:58:33.

### Outputs

    stats.json                                  every statistic, nested
    verification.csv                            computed values against the paper's, the audit's and the notebook's printed values
    summary.txt                                 plain-text summary; states the clock and window; ends with the applicable reference values that did not match
    annual_energy_TWh.csv, fleet_efficiency_mean_by_year_W_per_TH.csv,
    hashrate_mean_by_year_EHs.csv, scenario_summary.csv, linearity_cell22.csv,
    oos_monthly_ratio_2025.csv, hindcast_monthly_ratio.csv
    figure4.png ... figure9.png

`--compare` writes `comparison.csv` and `comparison.txt`: every numeric or date
statistic of two `stats.json` files, with the change. It refuses, with exit
status 2, to compare two outputs whose `clock` or `hindcast_window` differ. An
output written before these settings existed has neither key, so it counts as
different. `--allow-mixed-definitions` makes the comparison anyway. It then
prints a warning and writes the differing settings at the top of
`comparison.txt`.

If a calibration grid file is missing (`data/results/sensitivity_grid_results_rmsle_2018_2024_blocks.csv`
or `data/results/grid_2026-10-rerun/phase2_optimum.json`), the script prints a
warning naming the file, and the reference read from it is not checked.

### How the figures differ from the notebook's

- The font is Liberation Serif, with STIX for mathematics. The notebook used
  matplotlib's default, DejaVu Sans.
- Each figure keeps the notebook's width of 14.5 in. Its height comes from the
  aspect ratio of the picture beside the caption "Figure N." in the manuscript
  (`wp:extent`), as in `calibration/plot_grid_figure3.py`. `--size-in N W H`
  sets figure N's size directly. The figure is saved without
  `bbox_inches="tight"`, which would change the aspect ratio. So the legends
  that the notebook placed above (and, in Figure 4, below) the figure now sit
  inside it, and the axes are laid out to leave room for them.
- Figure 9's y label reads "($/MWh)". The notebook wrote `($/MWh$)`, which
  matplotlib read as mathematics, so the published label shows "/MWh" in italics
  and no dollar sign.
- Figure 9's break-even line is drawn at the set's electricity cost: $50/MWh for
  the original set, as in the notebook, and $40/MWh for the new runs.
- Figure 7 has no "Historical" legend entry, and its legend has four columns
  instead of five. The published figure has that entry with no line, because
  cell 23 selects historical rows with
  `2026-01-01 <= Hist_Timestamp <= 2025-11-08 17:58:33`, which no row satisfies.
  The script draws neither the empty line nor its legend entry.
- With the default `--clock historical`, Figures 4 and 5 plot both series
  against `Hist_Timestamp`. The notebook plotted them against `Timestamp`.
