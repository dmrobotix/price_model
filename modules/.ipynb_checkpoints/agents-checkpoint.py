# modules/agents.py
"""
Agent-based machine fleet module.

Each agent represents a single mining machine (one unit of an ASIC model).
The module provides:
  - creation of an "agents" DataFrame from the machine table
  - per-agent profitability calculations
  - turning machines on/off based on profitability
  - a greedy calibrator to match a target (historical) hashrate within tolerance
  - a forecast routine that turns machines on/off until break-even

Units & conventions:
  - hashrate columns: 'hashrate_th' (TH/s) and 'hashrate_hs' (hash/s)
  - efficiency: J / hash in 'eff_j_per_hash'
  - power: Watts in 'power_w'
  - daily revenues expressed in $/day (86400 seconds)
  - operating cost by default is electricity only; optionally scaled by ELEC_FRACTION for total operating
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from config import MACHINE_DATA_FILE, ELEC_FRACTION  # reuse same config keys
import logging

# Constants
SECONDS_PER_DAY = 86400.0
HASHES_PER_TH = 1e12
J_TO_WH = 1.0 / 3600.0  # 1 J/s = 1 W ; converting J/hash to Wh/hash uses block time; not used directly here

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Agent DataFrame helpers
# ---------------------------------------------------------------------
def load_machine_table(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load machine catalog CSV and normalize expected columns.
    Returns DataFrame with at least:
      - 'Model' (or 'Name')
      - 'Release' (datetime or string)
      - 'Power (W)'
      - 'Hashrate (TH/s)'
    """
    path = file_path or MACHINE_DATA_FILE
    df = pd.read_csv(path)
    # Normalise column names if older variants exist
    rename_map = {}
    if 'Power (W)' not in df.columns and 'Power_W' in df.columns:
        rename_map['Power_W'] = 'Power (W)'
    if 'Hashrate (TH/s)' not in df.columns and 'Hashrate_THs' in df.columns:
        rename_map['Hashrate_THs'] = 'Hashrate (TH/s)'
    if rename_map:
        df = df.rename(columns=rename_map)
    if 'Hashrate (TH/s)' not in df.columns or 'Power (W)' not in df.columns:
        raise ValueError("Machine table must contain 'Hashrate (TH/s)' and 'Power (W)' columns")
    return df.copy()


def initialize_agents(
    machine_data_file: Optional[str] = None,
    fleet_profile: Optional[Dict[str, int]] = None,
    default_count: int = 1,
    scale_to_hashrate: Optional[float] = None
) -> pd.DataFrame:
    """
    Create an 'agents' DataFrame where each row is a single machine-unit agent.

    Parameters
    ----------
    machine_data_file: Optional[str]
        Path to machine CSV. If None uses MACHINE_DATA_FILE from config.
    fleet_profile: Optional[dict]
        Mapping {model_identifier: count}. model_identifier matches a column 'Model' or 'Name'.
        If omitted, create `default_count` agents for each machine model found.
    default_count: int
        Number of units to create per model when fleet_profile is None.
    scale_to_hashrate: Optional[float]
        If provided, scale counts proportionally so that the total initial hashrate roughly matches
        `scale_to_hashrate` (in hash/s). Useful for matching INITIAL_HASHRATE.

    Returns
    -------
    agents_df : pd.DataFrame
      Columns include:
        - agent_id (int), model, release, hashrate_th, hashrate_hs, power_w, eff_j_per_hash,
          status_on (bool), last_profit_USD_per_day, operating_cost_USD_per_day
    """
    mdf = load_machine_table(machine_data_file)
    # Identify model/name column
    model_col = None
    for c in ('Model', 'Name', 'model'):
        if c in mdf.columns:
            model_col = c
            break
    if model_col is None:
        # fall back to index
        mdf = mdf.reset_index().rename(columns={'index': 'Model'})
        model_col = 'Model'

    rows = []
    for _, row in mdf.iterrows():
        model_name = row[model_col]
        count = fleet_profile.get(model_name, default_count) if fleet_profile else default_count
        for k in range(int(count)):
            rows.append({
                'model': model_name,
                'release': row.get('Release', pd.NaT),
                'hashrate_th': float(row['Hashrate (TH/s)']),
                'hashrate_hs': float(row['Hashrate (TH/s)']) * HASHES_PER_TH,
                'power_w': float(row['Power (W)']),
                # J / hash = (W per TH) / 1e12 ; compute via power/hashrate
                'eff_j_per_hash': (float(row['Power (W)']) / (float(row['Hashrate (TH/s)']) if float(row['Hashrate (TH/s)'])>0 else 1.0)) / HASHES_PER_TH,
                'status_on': False,
                'last_profit_USD_per_day': np.nan,
                'operating_cost_USD_per_day': np.nan
            })

    agents = pd.DataFrame(rows)
    if agents.empty:
        raise RuntimeError("No agents created; check machine file and fleet_profile.")
    # assign unique agent ids
    agents.insert(0, 'agent_id', range(1, len(agents) + 1))

    # Optionally scale counts to achieve a target total hashrate
    if scale_to_hashrate is not None:
        current = agents['hashrate_hs'].sum()
        if current <= 0:
            logger.warning("Cannot scale agents: current total hashrate is zero")
        else:
            scale_ratio = scale_to_hashrate / current
            # scale each agent's "replicated" count by rounding; easiest approach:
            # compute new counts proportional to hashrate contribution per model.
            # We'll produce a new fleet by integer replication of models proportional to scale_ratio.
            # To avoid exploding counts, only apply when scale_ratio > 1.0
            if scale_ratio > 1.0:
                # Determine per-model weights and rebuild
                agents_by_model = agents.groupby('model')['hashrate_hs'].sum()
                target_by_model = (agents_by_model / agents_by_model.sum() * scale_to_hashrate)
                new_rows = []
                for model, target_h in target_by_model.items():
                    # get representative row
                    prot = agents[agents['model'] == model].iloc[0]
                    unit_hash = prot['hashrate_hs']
                    n_units = max(1, int(round(target_h / unit_hash)))
                    for k in range(n_units):
                        new_rows.append(prot.to_dict())
                agents = pd.DataFrame(new_rows).reset_index(drop=True)
                agents.insert(0, 'agent_id', range(1, len(agents) + 1))

    return agents


# ---------------------------------------------------------------------
# Profitability & operating cost computations
# ---------------------------------------------------------------------
def calc_expected_revenue_per_hash(R_block: float, target: float) -> float:
    """
    R_t = (R_block * target) / 2**256  (BTC / hash)
    """
    return (R_block * target) / 2**256


def update_agent_profitability(
    agents: pd.DataFrame,
    R_block: float,
    target: float,
    P_USD: float,
    C_elec: float,
    total_operating: bool = False
) -> pd.DataFrame:
    """
    Update each agent's operating cost and last_profit_USD_per_day.

    - daily_rev_USD_per_agent = R_t [BTC/hash] * hashrate_hs * 86400 * P_USD
    - operating_cost_USD_per_day = (power_w * 24 / 1e6) * C_elec  (MWh/day * $/MWh)
      if total_operating True, operating_cost_USD_per_day /= ELEC_FRACTION

    Returns updated agents DataFrame (copy).
    """
    agents = agents.copy()
    R_t = calc_expected_revenue_per_hash(R_block, target)
    # vectorized daily revenue per agent
    agents['daily_rev_USD'] = R_t * agents['hashrate_hs'] * SECONDS_PER_DAY * P_USD
    # operating cost in $/day
    agents['operating_cost_USD_per_day'] = (agents['power_w'] * 24.0 / 1e6) * C_elec
    if total_operating:
        agents['operating_cost_USD_per_day'] = agents['operating_cost_USD_per_day'] / ELEC_FRACTION
    agents['last_profit_USD_per_day'] = agents['daily_rev_USD'] - agents['operating_cost_USD_per_day']
    return agents


# ---------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------
def aggregate_hashrate(agents: pd.DataFrame, only_on: bool = True) -> float:
    """
    Return total hashrate (hash/s). If only_on True, sum only agents with status_on True.
    """
    if only_on:
        return agents.loc[agents['status_on'], 'hashrate_hs'].sum()
    return agents['hashrate_hs'].sum()


# ---------------------------------------------------------------------
# Calibrate to historical hashrate (greedy)
# ---------------------------------------------------------------------
def calibrate_agents_to_target(
    agents: pd.DataFrame,
    target_hashrate: float,
    tolerance: float = 0.01,
    policy: str = 'profitability'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Greedy procedure to turn agents on/off to bring aggregate on-hashrate within tolerance of
    target_hashrate. The routine considers current profitability (agents['last_profit_USD_per_day'])
    and flips agents until the absolute relative difference is <= tolerance.

    Parameters
    ----------
    agents: pd.DataFrame
      Must already have computed 'last_profit_USD_per_day' via update_agent_profitability.
    target_hashrate: float (hash/s)
    tolerance: relative tolerance (e.g., 0.01 -> 1%)
    policy: 'profitability' (default) or 'efficiency'
      - 'profitability': prefer turning on agents with highest profit, turn off lowest-profit first
      - 'efficiency': prefer enabling most efficient (lowest J/hash) machines first

    Returns
    -------
    agents_updated, info_dict
      info_dict contains keys: final_hashrate, locked (bool), steps (int), final_rel_err
    """
    if 'last_profit_USD_per_day' not in agents.columns:
        raise ValueError("Agents must include 'last_profit_USD_per_day'. Call update_agent_profitability first.")

    # Work on a copy
    a = agents.copy().sort_values(['last_profit_USD_per_day', 'eff_j_per_hash'], ascending=[False, True]).reset_index(drop=True)
    current = aggregate_hashrate(a, only_on=True)
    if target_hashrate == 0:
        target_hashrate = 0.0
    rel_err = abs(current - target_hashrate) / max(target_hashrate, 1.0)
    steps = 0

    # if already within tolerance, lock and return
    if rel_err <= tolerance:
        return a, {'final_hashrate': current, 'locked': True, 'steps': 0, 'final_rel_err': rel_err}

    # Define ordering for on/off operations
    # To increase hashrate: turn on best candidates (by profit or efficiency)
    # To decrease hashrate: turn off worst performing agents (lowest profit)
    if policy == 'profitability':
        # descending profit ordering used for turning on
        turn_on_order = a.sort_values('last_profit_USD_per_day', ascending=False).index.tolist()
        turn_off_order = a.sort_values('last_profit_USD_per_day', ascending=True).index.tolist()
    elif policy == 'efficiency':
        turn_on_order = a.sort_values('eff_j_per_hash', ascending=True).index.tolist()
        turn_off_order = a.sort_values('eff_j_per_hash', ascending=False).index.tolist()
    else:
        raise ValueError("Unknown policy")

    # Greedy loop with safety cap
    max_steps = len(a) * 5
    # Start with current statuses; prefer turning on profitable machines if under target
    while steps < max_steps:
        current = aggregate_hashrate(a, only_on=True)
        rel_err = abs(current - target_hashrate) / max(target_hashrate, 1.0)
        if rel_err <= tolerance:
            break
        # decide direction
        if current < target_hashrate:
            # need more hashrate -> turn on the next best candidate that's currently off
            candidate_idx = next((i for i in turn_on_order if not bool(a.at[i, 'status_on'])), None)
            if candidate_idx is None:
                # nothing else to turn on
                break
            a.at[candidate_idx, 'status_on'] = True
        else:
            # current > target -> turn off the worst candidate that's currently on
            candidate_idx = next((i for i in turn_off_order if bool(a.at[i, 'status_on'])), None)
            if candidate_idx is None:
                break
            a.at[candidate_idx, 'status_on'] = False
        steps += 1

    final = aggregate_hashrate(a, only_on=True)
    final_rel_err = abs(final - target_hashrate) / max(target_hashrate, 1.0)
    locked = final_rel_err <= tolerance
    return a, {'final_hashrate': final, 'locked': locked, 'steps': steps, 'final_rel_err': final_rel_err}


# ---------------------------------------------------------------------
# Forecasting-driven toggle until break-even
# ---------------------------------------------------------------------
def forecast_until_breakeven(
    agents: pd.DataFrame,
    R_block: float,
    target: float,
    P_USD: float,
    C_elec: float,
    total_operating: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Turn machines on (greedily) until the next candidate to turn on would have non-positive profit.
    Also turn off clearly unprofitable machines.

    Strategy:
      1) Update per-agent profitability.
      2) Turn off any currently-on machines with negative profit (immediate).
      3) Sort remaining off-machines by last_profit_USD_per_day desc, turn them on while their profit>0.
      4) Stop when the next candidate has profit <= 0.

    Returns updated agents and info dict (final_hashrate, n_turned_on, n_turned_off).
    """
    agents = update_agent_profitability(agents, R_block, target, P_USD, C_elec, total_operating)
    a = agents.copy()

    # 1) turn off currently-on unprofitable machines
    on_mask = a['status_on']
    to_turn_off = a.index[on_mask & (a['last_profit_USD_per_day'] <= 0)].tolist()
    a.loc[to_turn_off, 'status_on'] = False

    # 2) candidates to turn on (off machines with profit > 0), sorted by profit desc
    off_candidates = a[~a['status_on']].copy()
    off_candidates = off_candidates.sort_values('last_profit_USD_per_day', ascending=False)

    turned_on = 0
    for idx, row in off_candidates.iterrows():
        if row['last_profit_USD_per_day'] > 0:
            a.at[idx, 'status_on'] = True
            turned_on += 1
        else:
            # next candidate is not profitable -> stop
            break

    turned_off = len(to_turn_off)
    final_hash = aggregate_hashrate(a, only_on=True)
    return a, {'final_hashrate': final_hash, 'n_turned_on': turned_on, 'n_turned_off': turned_off}


# ---------------------------------------------------------------------
# Small utility / debug driver
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test if run directly; not a formal unit test.
    print("Agents module quick smoke test.")
    try:
        agents = initialize_agents(default_count=1)
        print(f"Created {len(agents)} agents. Total fleet hash (TH/s): {agents['hashrate_th'].sum():.2f}")
        # sample R_block/target and price: use small numbers so profit likely negative -> mostly off
        R_block = 6.25
        # naive default target (will be replaced in simulation)
        sample_target = 1.0
        P_USD = 30000.0
        C_elec = 50.0
        agents = update_agent_profitability(agents, R_block, sample_target, P_USD, C_elec, total_operating=False)
        print("Sample profit stats: min, median, max (USD/day):",
              agents['last_profit_USD_per_day'].min(),
              agents['last_profit_USD_per_day'].median(),
              agents['last_profit_USD_per_day'].max())
        # forecast until breakeven
        agents_on, info = forecast_until_breakeven(agents, R_block, sample_target, P_USD, C_elec)
        print("Forecast result:", info)
    except Exception as e:
        print("Smoke test failed:", e)
        raise
