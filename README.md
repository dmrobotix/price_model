# PRICE: Protocol Revenue and Integrated Cryptocurrency Energy Model

This repository holds the code of the PRICE model and the scripts that produce the
results of the accompanying paper. PRICE is a block-level simulator of the Bitcoin
network. It is used to estimate network hashrate, energy use and miner revenue, over
history and in forecast scenarios.

## What the model does

The simulation advances one block at a time, from the genesis block onward. At each
block it does the following:

1. It computes the miners' expected revenue per unit of energy from the block subsidy,
   the transaction fees, the bitcoin price, the difficulty and the fleet efficiency.
2. It maps that revenue to an expected block time through a constant-elasticity supply
   model. The elasticity is `S` (from 2018 on) or `S_0` (before 2018). The electricity
   cost is `C_ELEC` (from 2018 on) or `C_ELEC_0` (before 2018).
3. It sets each block time to the expected block time from step 2 (the model is
   deterministic; the random draw in the code is switched off) and updates the difficulty
   every 2,016 blocks, as Bitcoin Core does. The simulated hashrate follows from the difficulty and the block times.
4. It computes the fleet efficiency (J/TH) once per calendar day with the CBECI
   method. The machine table gives each machine's release date and efficiency. Each
   machine is deployed two months after release and retired five years after
   deployment. Only machines that are profitable at the electricity cost are counted,
   and they are weighted by age.
5. It computes power and energy from the hashrate and the efficiency.

Up to the historical cutoff (`historical_cutoff` in `main.py`), the block timestamps
come from the historical block data. After it, the economic model sets the block times.
The historical price and fee data are used up to the last timestamp of the
transaction-fee file. After that timestamp, the price follows a forecast and the fee of
each block is 4% of its subsidy. The fleet efficiency follows one of two scenarios, set
by `EFFICIENCY_SCENARIO` in `config.py`:

- `frozen`: the fleet is drawn from the machines in the machine table only;
- `frontier`: the fleet also includes one synthetic machine. Its efficiency follows a
  power law fitted to the lowest energy per hash (J/TH) reached by each release date. It
  enters after the last fee timestamp, once the most recently released machine in the
  table has been deployed.

## Files

| Path | Contents |
|---|---|
| `main.py` | Runs one simulation and writes its result CSV. |
| `config.py` | Model constants, calibrated parameters, input file paths, and the `PRICE_*` environment overrides. |
| `modules/` | The model: `simulation.py` (block loop), `network.py` (difficulty and hashrate), `economics.py` (revenue to block time), `efficiency_cbeci.py` (CBECI fleet efficiency and the frontier fit), `energy.py`, `energy_cbeci.py`, `price.py` (historical prices and forecasts), `data_processing.py` (input loaders and error metrics), `visualization.py`, `debug.py`. |
| `validation/` | The CBECI replication (`cbeci_replication.py`), its figure (`plot_cbeci_replication.py`, Figure 2), and the outputs of the run reported in the paper (`results/`). |
| `calibration/` | The driver of the two calibration grids (`run_grid.py`, `run_grid_point.py`) and their figure (`plot_grid_figure3.py`, Figure 3). See `calibration/README.md`. |
| `notebooks/` | The two calibration grid scripts that `calibration/run_grid_point.py` imports. |
| `scenarios/` | The run definitions (`runs.json`), the scenario driver (`run_scenarios.py`, `run_one.py`), the reproduction check (`prepare_april_reproduction.py`, `verify_reproduction.py`) and the statistics and Figures 4 to 9 (`analyze_results.py`). See `scenarios/README.md`. |
| `REQUIREMENTS.txt` | Package versions used for the paper. |

## Requirements

The paper's runs used Python 3.11.15 on Linux, in two environments built with the
same pinned package versions, which are listed in `REQUIREMENTS.txt`:

    pip install -r REQUIREMENTS.txt

The drivers in `calibration/` and `scenarios/` import the `fcntl` and `resource`
modules of the Python standard library. Those modules exist only on Unix-like systems.

The figure scripts use the Liberation Serif font. `calibration/plot_grid_figure3.py`
and `scenarios/analyze_results.py` stop if it is not installed.
`validation/plot_cbeci_replication.py` falls back to DejaVu Serif if Liberation Serif
is not installed.

## Input data

The input files are not in this repository. The `data/` directory is excluded by
`.gitignore`.

### Where the code looks for them

Every input and output path in `config.py`, `main.py` and `modules/debug.py` starts
with `../data/`. These paths are relative to the current working directory, so the
inputs must be in a directory named `data` inside the parent of the working directory.
`main.py` also writes its log to `../data/logs/model_debug.log`, and that directory
must exist before the run starts.

Two layouts work:

- **`data/` at the repository root.** Run from a direct subdirectory of the repository,
  for example `cd notebooks && python ../main.py`. The validation and calibration
  scripts use this layout: each one changes into its own directory (`validation/` or
  `notebooks/`) before it loads data.
- **`data/` beside the repository.** Run from the repository root with
  `python main.py`. The inputs are then read from `<parent of the repository>/data/`.

`scenarios/run_scenarios.py` does not use either layout directly. It takes the inputs
from `--data-dir` and creates a separate working directory for every run (see
"Scenario runs" below).

`data/` may be a symbolic link. `.gitignore` excludes both a directory and a link of
that name at the repository root.

### Files each workflow reads

| Workflow | Files read from `data/` |
|---|---|
| `main.py` with no `PRICE_*` variables set | `combined_block_data_latest.csv`, `market_price_min_latest.csv`, `txfee_data.csv`, `cbeci_machines_090325.csv` |
| CBECI replication (`validation/cbeci_replication.py`) | the same four files as `main.py` (the `config.py` defaults) |
| Calibration grids (`calibration/run_grid.py`) | `combined_block_data_latest.csv`, `market_price_min.csv`, `txfee_data.csv`, `cbeci_machines_090325.csv`. The price file is named in the two grid scripts in `notebooks/`. The other three come from `config.py`. |
| The ten scenario runs and the reproduction run (`scenarios/runs.json`) | `combined_block_data.csv`, `market_price_min.csv`, `txfee_data.csv`, `cbeci_machines_090325.csv` |

The scenario runs override the block and price files of `config.py` with the two older
files, which are the files the paper's original runs used. `scenarios/README.md`
("Input files") gives the evidence. The two block files are identical over the blocks
they share in `Block_Time_Seconds` and `Bits`. The two price files are identical over
the minutes they share.

### Size, coverage and origin of each file

| File | Bytes | Last row | Origin |
|---|---|---|---|
| `cbeci_machines_090325.csv` | 11,170 | 166 machines | The machine table published on the CBECI website (Cambridge Centre for Alternative Finance, https://ccaf.io/cbnsi/cbeci), as stated in the paper's Data Sources section. The download date is not recorded in this repository. |
| `txfee_data.csv` | 87,084,626 | block 922,755, 2025-11-08 17:58:32 UTC | Per-block fee statistics (`Height`, `timestamp` in milliseconds, `total_fees` and others). The paper states that block-level data, including transaction fees, were collected with Bitcoin Core through custom Python scripts. The script that wrote this file is not in this repository. The last fee timestamp sets the start of the forecast. |
| `combined_block_data.csv` | 176,547,088 | block 925,641, 2025-11-29 00:32:16 | Per-block height, hash, timestamp, inter-block interval, `Bits` and target. Collected with Bitcoin Core (RPC) and the mempool.space REST API by scripts that are not in this repository. |
| `combined_block_data_latest.csv` | 180,410,516 | block 936,248, 2026-02-12 17:19:49 | The same source and columns, extended to a later block, with two extra columns. |
| `market_price_min.csv` | 238,177,951 | 2025-11-27 18:59:00 | BTC-USD minute closing prices from the CoinDesk Data API, joined to daily prices from blockchain.com (`api.blockchain.info/charts/market-price`) for the period before the minute series begins. The 560 rows before 2010-07-17 are daily rows with price 0.0, because no market price exists for that period. The download and merge scripts are not in this repository. |
| `market_price_min_latest.csv` | 242,253,728 | 2026-02-12 12:28:00 | The same sources, extended to a later date. |

SHA-256 of the files used for the paper:

    ee5e360ffc1b665c43b89bb32b25ec91fac963a8b8eddd7cdceedc6efc39816a  cbeci_machines_090325.csv
    470bae86638000484b5af446b4b1cbccbb2569fd3a7875d5e2230f47bb9a2240  txfee_data.csv
    ecb9d3142055a70d669dc1cf2529ef55ff26550522df55a03b30a158823df290  combined_block_data.csv
    6f8f800dc26ef801b788e1fdd53c8ba3ba3442eefcd3d4fb0012f59378a450c0  combined_block_data_latest.csv
    3ceb2550a4ccf404f83c7fd9b4eaddc1e21fa9456067bf4178f0bf64fa96abe9  market_price_min.csv
    f603234721637ee3223d96e9bd5c591287ef0c54bb7c9c854ce8819f6ed7f8f4  market_price_min_latest.csv

Two workflows also read outputs of earlier runs:

- `scenarios/verify_reproduction.py` compares a re-run with the saved original output
  `simulation_results_efficiency_frozen_price_fixed.csv` (560,587,038 bytes).
- `scenarios/analyze_results.py --set original` reads the ten original run CSVs named
  in the `original` field of `scenarios/runs.json`. Each is between 426,773,748 and
  566,706,788 bytes.

These outputs are not in this repository either.

## Running `main.py`

With the inputs in place (see "Where the code looks for them"):

    mkdir -p data/logs
    cd notebooks && python ../main.py

With no `PRICE_*` environment variables set, `main.py` simulates 921,683 blocks with
the historical cutoff at 2025-01-01 00:00:00. These are the settings of
the out-of-sample test (the `testing` run in `scenarios/runs.json`). The run differs
from the `testing` run in two ways. First, it reads the `_latest` block and price files
named in `config.py`. Second, it writes its output to
`../data/simulation_results_hindcasting.csv`, because that is the file name set in
`main.py`.

`main.py` writes three files:

- `../data/<file_name>.csv`, one row per block;
- `../data/hashrate_comparison.png`;
- `../data/logs/model_debug.log`.

It also prints a comparison of the simulated annual energy with CBECI's annual
estimates.

### Settings without editing the code

`config.py` and `main.py` read nine optional environment variables:

| Variable | Overrides |
|---|---|
| `PRICE_EFFICIENCY_SCENARIO` | `EFFICIENCY_SCENARIO` (`frozen` or `frontier`) |
| `PRICE_FORECAST_MODEL` | `FORECAST_MODEL` (`powerlaw`, `fixed`, `constant`, `linear`, `logistic`) |
| `PRICE_FORECAST_TARGET_DATE` | `FORECAST_TARGET_DATE` |
| `PRICE_FORECAST_TARGET_PRICE` | `FORECAST_TARGET_PRICE` (a number, or `none`) |
| `PRICE_BLOCK_PACE_DATA` | `BLOCK_PACE_DATA` (a file name in `../data/`) |
| `PRICE_PRICE_DATA` | `PRICE_DATA` (a file name in `../data/`) |
| `PRICE_NUM_STEPS` | `num_steps` in `main.py` |
| `PRICE_HISTORICAL_CUTOFF` | `params['historical_cutoff']` in `main.py` |
| `PRICE_OUTPUT_NAME` | `file_name` in `main.py` |

With none of them set, every value is the one written in the code. A malformed value
raises `ValueError` when `config.py` is imported. Any other variable whose name starts
with `PRICE_` also raises `ValueError`. `scenarios/README.md` gives the accepted
formats.

For example, this command runs the forecast scenario with frontier efficiency and a
price of $1,000,000 in 2032 on the original input files:

    cd notebooks && \
    PRICE_EFFICIENCY_SCENARIO=frontier PRICE_FORECAST_MODEL=powerlaw \
    PRICE_FORECAST_TARGET_DATE=2032-01-01 PRICE_FORECAST_TARGET_PRICE=1000000 \
    PRICE_BLOCK_PACE_DATA=combined_block_data.csv PRICE_PRICE_DATA=market_price_min.csv \
    PRICE_NUM_STEPS=1260000 PRICE_HISTORICAL_CUTOFF=2025-11-08T17:58:32 \
    PRICE_OUTPUT_NAME=simulation_results_efficiency_frontier_price_1M \
    python ../main.py

`scenarios/run_scenarios.py` sets these variables from `scenarios/runs.json` and
checks the result, so it is the recommended way to run the paper's scenarios.

## CBECI replication (Figure 2)

`validation/cbeci_replication.py` runs the model over the historical blocks to the end
of 2023 with CBECI's own assumptions. These are an electricity price of $50/MWh in every
year and a PUE of 1.10. It compares the resulting annual electricity use with CBECI's
published annual estimates for 2011 to 2023. Run it with `data/` at the repository root:

    MPLBACKEND=Agg python validation/cbeci_replication.py [--out-dir DIR]

The script changes into `validation/` itself. The run takes about 40 to 50 minutes. The
outputs are `annual_comparison.csv`, `error_summary.csv`,
`daily_efficiency_J_per_TH.csv` and `model_debug.log`. They are written to
`data/results/cbeci_replication/`, or to `--out-dir`. A relative `--out-dir` is resolved
against the repository root. The script refuses to overwrite existing outputs.

The CBECI manufacturer filter is applied by the machine loader in
`modules/efficiency_cbeci.py`, so it applies to every run of the model. From July 2014
on, the filter keeps only Bitmain, MicroBT and Canaan hardware.

`validation/plot_cbeci_replication.py` draws Figure 2 from `annual_comparison.csv`:

    python validation/plot_cbeci_replication.py --in-dir data/results/cbeci_replication \
        --size-in 6.9250 3.5243

Its default `--in-dir` (`data/results/cbeci_replication_v3`) is the name of the
authors' run, so pass `--in-dir`. See "Paths that refer to the authors' files" below for
`--size-in`.

## Calibration grids (Figure 3)

The parameters `S_0`, `C_ELEC_0`, `S` and `C_ELEC` in `config.py` are the minima of two
grid searches of the RMSLE between simulated and historical hashrate:

- the pre-2018 grid varies `S_0` and `C_elec_0` over blocks 68,633 to 501,994;
- the 2018-2024 grid varies `S` and `C_elec` over blocks 501,995 to 877,279, with `S_0`
  and `C_elec_0` fixed at the optimum of the first grid.

The grid scripts are `notebooks/sensitivity_analysis_pre2018_blocks.py` and
`notebooks/sensitivity_analysis_2018-2024_rmsle_blocks.py`. Run directly from
`notebooks/`, each script evaluates 165 points serially (elasticity 0.00 to 0.10 in
steps of 0.01, times electricity cost 10 to 150 $/MWh in steps of 10) and writes a CSV
into `notebooks/`. `calibration/run_grid.py` evaluates the same points one per process,
in parallel and resumably. It leaves out elasticity 0 unless `--include-s0` is given,
which leaves 150 points per grid. It runs the 2018-2024 grid with `S_0` and `C_elec_0`
fixed at the optimum of the pre-2018 grid:

    python calibration/run_grid.py --out-dir <new directory> --jobs 4 --python "$(which python)"

Each point simulates 888,300 blocks and needs about 2.2 GB of memory.
`calibration/README.md` describes the phases, the outputs, resuming, and the exit
status.

`calibration/plot_grid_figure3.py` draws Figure 3 from the two merged grid CSVs:

    python calibration/plot_grid_figure3.py --pre2018 <out-dir>/grid_pre2018.csv \
        --modern <out-dir>/grid_2018_2024.csv --out <out-dir>/figure3.png \
        --size-in 16.0000 5.2756

## Scenario runs (Figures 4 to 9)

`scenarios/runs.json` defines the ten runs of the paper:

- `testing`: the out-of-sample test, 921,683 blocks, economic model from 2025-01-01;
- `hindcasting`: the hindcast of Figure 4, 877,280 blocks, economic model from
  2010-07-17;
- eight forecast scenarios, 1,260,000 blocks each (to about 2032), from 2025-11-08
  17:58:32. They combine two efficiency scenarios (`frozen`, `frontier`) with four price
  paths (`fixed` at the last historical price, and power laws reaching $200,000,
  $500,000 and $1,000,000 on 2032-01-01).

All ten read the original input files `combined_block_data.csv` and
`market_price_min.csv`. Run them with:

    python scenarios/run_scenarios.py --out-dir <new directory> --data-dir <directory with the four inputs> \
        --jobs 2 --python "$(which python)"

The driver starts one `scenarios/run_one.py` process per run. Each run gets its own
directory `<out-dir>/runs/<name>/`. Its working directory is `work/`, and `data/` beside
it holds links to the input files and receives the run's outputs. So `main.py` finds
`../data/` without any change to the code. The driver passes each run's settings as the
`PRICE_*` variables. Before the simulation, `run_one.py` checks that the values in
effect equal the requested ones. After it, `run_one.py` checks the CSV and writes
`DONE.json`. Budget about 4 GB of memory per run. `--runs` selects runs by name.
`--child-num-steps N` (hidden from `--help`) shortens every run to N blocks for
testing. 30,000 blocks is known to work.

The settings of each run were read from the saved original outputs. `scenarios/README.md`
gives that evidence, the output layout, resuming, and signals.

### Reproduction check

The reproduction check re-runs one original scenario (`frozen_price_fixed`) with the
April 2026 code and compares the output with the saved original, row by row.
`scenarios/prepare_april_reproduction.py` exports the April code from a git tag and
patches its hardcoded settings. `scenarios/verify_reproduction.py` compares the two CSVs.
`scenarios/README.md` ("Reproduction check") gives the steps.

This check cannot be run from this repository alone. `prepare_april_reproduction.py`
reads the tag `paper3-april-2026` of the authors' working repository, and this
repository does not have that tag. This repository's April commit `fbe8309` is not a
substitute. Its `config.py` sets `EFFICIENCY_SCENARIO = "frozen"`, where the tagged
April code sets `"frontier"`, so the script stops before patching anything. The check
also needs the saved original output, which is not published.

### Statistics and figures

`scenarios/analyze_results.py` computes the paper's statistics and draws Figures 4 to 9
from the ten CSVs of one set of runs:

    python scenarios/analyze_results.py --set new --runs-dir <out-dir>/runs \
        --out-dir <new directory> --size-in 4 14.5 7.9542 --size-in 5 14.5 6.9388 \
        --size-in 6 14.5 5.0587 --size-in 7 14.5 5.0689 --size-in 8 14.5 5.0587 \
        --size-in 9 14.5 5.0660

`--no-figures` computes the statistics only. `--set original` reads the original run
CSVs from `data/results/`. For that set, pass `--c-elec 50`. Without it, the script
reads `C_ELEC` from `config.py` at the tag `paper3-april-2026`, which this repository
does not have. `--compare` compares two output directories. `scenarios/README.md`
("Statistics and Figures 4 to 9") describes every statistic and option.

## Paths that refer to the authors' files

Some defaults in the scripts name files that exist only in the authors' working
repository. None of them changes a computed value. Each one needs an option in this
repository:

- `validation/plot_cbeci_replication.py`, `calibration/plot_grid_figure3.py` and
  `scenarios/analyze_results.py` read the figure sizes from the manuscript's `.docx`,
  at a path on the authors' machine (`DEFAULT_DOCX`). Pass `--size-in` instead. The
  sizes in this README were read from the manuscript draft of 4 October 2026 by the
  scripts' own functions. They are, in inches: Figure 2 6.9250 by 3.5243; Figure 3
  16.0000 by 5.2756; Figures 4 to 9 14.5 wide by 7.9542, 6.9388, 5.0587, 5.0689,
  5.0587 and 5.0660.
- `validation/plot_cbeci_replication.py` defaults to `--in-dir
  data/results/cbeci_replication_v3`, and `calibration/plot_grid_figure3.py` defaults
  to `data/results/grid_2026-10-01/`. Both are names of the authors' runs.
- `scenarios/analyze_results.py --set new` defaults to
  `data/results/scenarios_2026-10/runs`. Pass `--runs-dir`. It also looks for two grid
  files in `data/results/` to check the calibration RMSLE. If they are missing, it
  prints a warning and skips that check.
- The docstrings refer to `data/results/scenario_results.ipynb`, the notebook that
  produced the original figures. That notebook is not published.
- `config.py` records the calibration grid run by its directory and a commit of the
  working repository (`09730b6`). That commit is not in this repository.
- `scenarios/README.md` and `calibration/README.md` describe the runs on the authors'
  machines (host names, paths and measured memory and time). The commands apply here
  with the paths changed.

## Changes since the April 2026 snapshot

Commit `fbe8309` is the snapshot published in April 2026. The changes since then are
these:

- `modules/efficiency_cbeci.py` reads machine release dates from
  `UNIX_date_of_release`, and otherwise parses `Date of release` day first. The April
  code parsed that column month first.
- The machine loader applies the CBECI manufacturer filter (Bitmain, MicroBT and
  Canaan from July 2014 on).
- The frontier efficiency curve is fitted in log space. If the fit fails, a frontier
  run stops with an error. The April code used the initial guess in that case.
- The daily efficiency aggregates every block stamped with a calendar day that has
  arrived by the time that day is aggregated (a block stamped with an earlier day that
  arrives later is not added), and adds one moving-average entry per day. Block timestamps are not monotonic. The April code could
  leave blocks of a day out of that day's aggregate, and could add a second
  moving-average entry for a day.
- `run_simulation` resets the per-run efficiency state, so that several runs in one
  process are independent.
- `config.py` holds the parameters of the re-run calibration grid: `S_0 = 0.09`,
  `C_ELEC_0 = 110`, `S = 0.02`, `C_ELEC = 40`. `EFFICIENCY_SCENARIO` is `"frontier"`.
- `config.py` and `main.py` read the `PRICE_*` environment overrides.
- The validation, calibration and scenario scripts and `REQUIREMENTS.txt` were added.

## License

This software is licensed under the PolyForm Noncommercial License 1.0.0 (see
`LICENSE.md`). It may be used, changed and shared for any noncommercial purpose,
including personal use, research, education, and use by charitable, educational,
public research and government organisations. Commercial use is not licensed under
these terms. To ask about a commercial license, contact the author.
