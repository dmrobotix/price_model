import copy
import math
import traceback
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import logging
from modules import network, economics, energy, debug
from modules.boundaries import GENESIS_TIME_S, ERA_SPLIT_UTC, boundary_block, utc_seconds
from modules.efficiency_cbeci import compute_dynamic_efficiency, reset_run_state
from modules.price import get_price_for_date, forecast_price
from modules.economics import apply_hashrate_shock
from config import EXPECTED_BLOCK_TIME, S as S_CONFIG, CALIBRATION_MODE, TX_FEE_PCT

def _advance_clock(state: dict) -> None:
    """Advance the simulated clock by state['T_block'] seconds without losing time.

    state['sim_clock_s'] is the exact clock (float seconds, UTC epoch). The block's
    integer timestamp state['sim_timestamp_s'] is the floor of it, so it is never more
    than 1 s behind the exact clock and the error does not accumulate. Before
    2026-10-05 the integer timestamp itself was advanced by int(T_block), which dropped
    the fractional part of every block time (about 0.7 s per block at T = 599.7 s).
    """
    state['sim_clock_s'] = float(state['sim_clock_s']) + float(state['T_block'])
    state['sim_timestamp_s'] = math.floor(state['sim_clock_s'])
    state['sim_timestamp'] = pd.to_datetime(state['sim_timestamp_s'], unit="s", utc=True).tz_convert(None)


def _block_subsidy(block_height: int, params: dict) -> float:
    """Protocol subsidy (BTC) of the block at block_height: the halving rule of simulate_step."""
    return params['initial_block_subsidy'] / (2 ** (block_height // params['halving_interval']))


def _history_subsidy(entry: dict, params: dict) -> float:
    """Subsidy of the block that a history entry records.

    history[i] is the state after block i was mined (run_simulation sets
    'block_height' on every entry it creates). The initial state, history[0], may
    carry no 'block_height'; then the subsidy it was constructed with is used.
    """
    if 'block_height' in entry:
        return _block_subsidy(int(entry['block_height']), params)
    return float(entry['R_block'])


def simulate_step(state: dict, params: dict, block_height: int, history: list) -> dict:
    """
    Perform one simulation step by updating the simulation state.
    """
    # Ensure required keys exist (sim_timestamp_s is now critical)
    for key in ('target', 'D', 'H', 'sim_timestamp', 'sim_timestamp_s'):
        if key not in state:
            raise ValueError(f"Missing required state key: '{key}'.")
    # Exact clock (float seconds); see _advance_clock. A state without it starts it
    # at the integer timestamp.
    state.setdefault('sim_clock_s', float(state['sim_timestamp_s']))

    try:
        # --- Halving rule ---
        state['R_block'] = _block_subsidy(block_height, params)

        # --- Time State Setup ---
        # 'now_s' is the timestamp (seconds) BEFORE the current block is mined.
        now_s = state['sim_timestamp_s']
        
        # 'now_dt' is the UTC-naive datetime for modules that require datetime inputs.
        # We explicitly convert from seconds to ensure consistency.
        now_dt = pd.to_datetime(now_s, unit="s", utc=True).tz_convert(None)
        
        # Determine cutoff in seconds
        cutoff_dt = params['historical_cutoff']
        # Convert cutoff to integer timestamp if it isn't already, assuming UTC-naive input
        if isinstance(cutoff_dt, (datetime, pd.Timestamp)):
            cutoff_s = int(cutoff_dt.replace(tzinfo=timezone.utc).timestamp())
        else:
            cutoff_s = cutoff_dt

        # --- Market price ---
        # Access historical block-level data (includes Total_Fees_BTC)
        bp = params['block_paces']
        # We use the integer timestamp for price lookup to avoid timezone ambiguity
        last_hist_time = params['last_hist_time'] 
        # Ensure last_hist_time is comparable (convert to seconds if needed for logic, 
        # though usually price_forecaster handles the type logic)
        
        if now_dt > last_hist_time:
            # Forecast region
            state['P_USD'] = params['price_forecaster'](now_dt)
        else:
            # Historical region: use fast streaming lookup with integer seconds
            state['P_USD'] = params['price_lookup'](now_s)

        # --- Transaction fees ---
        # Use historical fees as long as we are within the same cutoff used for P_USD.
        # This is intentionally independent of the historical-vs-forecast regime switch (cutoff_s).
        if (now_dt <= last_hist_time) and (block_height in bp.index):
            state['TX_fee'] = float(bp.loc[block_height, 'Total_Fees_BTC'])
        elif (now_s <= cutoff_s) and (block_height not in bp.index):
            # Historical-mode fallback if the block height is missing from bp
            state['TX_fee'] = 0.0
        else:
            # Forecast rule once price history is no longer available
            state['TX_fee'] = float(TX_FEE_PCT) * float(state['R_block'])

        # --- Determine block time ---
        cal_mode = params.get('calibration_mode', CALIBRATION_MODE)
        S_eff = params.get('S', S_CONFIG) if cal_mode else S_CONFIG

        # --- Era (S_0/C_elec_0 before, S/C_elec after) by block height ---
        # The era split is the 2018-01-01 UTC boundary (modules/boundaries.py): blocks up
        # to params['last_early_era_height'] are early era, later blocks modern era. The
        # split is by height so that a run whose simulated clock differs from the real
        # one changes era at the same block as the calibration windows.
        modern_era = block_height > params['last_early_era_height']

        # --- Gather daily data for CBECI (Optimization) ---
        if not cal_mode or (cal_mode and now_dt >= cutoff_dt):
            current_day = now_dt.date()
            n_hist = len(history)
            start_idx = params.get('daily_window_start_idx', 0)
            aggregated_day = params.get('daily_window_day', None)
            
            day_to_aggregate = None
            if aggregated_day is not None and current_day > aggregated_day:
                day_to_aggregate = aggregated_day
            
            if day_to_aggregate is not None:
                # Advance start_idx
                while start_idx < n_hist:
                    # Check history entry date. Safe to access 'sim_timestamp' from history
                    # as it was stored as a datetime object in previous steps.
                    hist_date = history[start_idx]['sim_timestamp'].date()
                    if hist_date >= day_to_aggregate:
                        break
                    start_idx += 1
                
                # Block timestamps are not monotonic: a block stamped day X+1 can be
                # followed by a block stamped day X. Scan the whole remainder
                # history[start_idx:n_hist] and keep every entry stamped
                # day_to_aggregate, rather than stopping at the first entry from
                # another day. The next window starts at the first entry stamped
                # after day_to_aggregate. Aggregation still runs only once
                # current_day > aggregated_day, so it uses blocks already produced
                # (causal); a block stamped day X that arrives after day X was
                # aggregated is not added to that aggregate.
                same_day_history = []
                end_idx = n_hist
                for i in range(start_idx, n_hist):
                    h_date = history[i]['sim_timestamp'].date()
                    if h_date == day_to_aggregate:
                        same_day_history.append(history[i])
                    elif h_date > day_to_aggregate and end_idx == n_hist:
                        end_idx = i
                
                if not same_day_history:
                    daily_difficulties = np.array([state['D']])
                    daily_block_times = np.array([state.get('T_block', EXPECTED_BLOCK_TIME)])
                    daily_tx_fees = np.array([state.get('TX_fee', 0.0)])
                    daily_subsidies = np.array([state['R_block']])
                else:
                    daily_difficulties = np.array([h['D'] for h in same_day_history])
                    daily_block_times = np.array([h.get('T_block', EXPECTED_BLOCK_TIME) for h in same_day_history])
                    daily_tx_fees = np.array([h.get('TX_fee', 0.0) for h in same_day_history])
                    # Each block's own subsidy, aligned 1:1 with daily_difficulties, so that
                    # a day containing a halving is priced at both subsidies. These are the
                    # history entries stamped day_to_aggregate (the day being closed; the block
                    # that opens current_day is history[-1]), i.e. exactly the blocks whose
                    # difficulty and fees are aggregated here.
                    daily_subsidies = np.array([_history_subsidy(h, params) for h in same_day_history])
                
                aggregate_daily_fees = float(np.sum(daily_tx_fees))
                
                params['daily_difficulties_cached'] = daily_difficulties
                params['daily_block_times_cached'] = daily_block_times
                params['daily_block_subsidies_cached'] = daily_subsidies
                params['aggregate_daily_fees_cached'] = aggregate_daily_fees
                params['daily_window_start_idx'] = end_idx 
                params['daily_window_day'] = current_day 
            
            daily_difficulties = params.get('daily_difficulties_cached', np.array([state['D']]))
            daily_block_times = params.get('daily_block_times_cached', np.array([state.get('T_block', EXPECTED_BLOCK_TIME)]))
            daily_subsidies = params.get('daily_block_subsidies_cached', np.array([state['R_block']]))
            aggregate_daily_fees = params.get('aggregate_daily_fees_cached', 0.0)
            state['aggregate_daily_fees'] = aggregate_daily_fees
            
            if params.get('daily_window_day') is None:
                params['daily_window_day'] = current_day 

        elif cal_mode and now_dt < cutoff_dt:
            state['aggregate_daily_fees'] = 0.0

        # --- Historical vs Forecast logic ---
        # Note: We compare integer seconds (now_s) with cutoff_s
        if now_s <= cutoff_s: 
            # HISTORICAL MODE
            # We must safeguard against block_height exceeding available history
            if block_height in bp.index:
                state['T_block'] = float(bp.loc[block_height, 'Inter_Block_Interval_Seconds'])
                
                # Update timestamp strictly from historical data
                hist_ts_s = int(bp.loc[block_height, "Block_Time_Seconds"])
                state["sim_timestamp_s"] = hist_ts_s
                state["sim_clock_s"] = float(hist_ts_s)
                # Sync datetime object (UTC-naive)
                state["sim_timestamp"] = pd.to_datetime(hist_ts_s, unit="s", utc=True).tz_convert(None)
            else:
                # Fallback if historical data is missing (should verify cutoff alignment)
                state['T_block'] = EXPECTED_BLOCK_TIME
                _advance_clock(state)

            if not cal_mode: 
                # --- Efficiency ---
                fix_flag = params.get('fix_efficiency', False)
                eff_cutoff = params['last_hist_time']
                
                if fix_flag and now_dt > eff_cutoff:
                    if params['fixed_efficiency'] is None:
                        params['fixed_efficiency'] = state['efficiency']
                    state['efficiency'] = params['fixed_efficiency']
                else:
                    state['efficiency'], state['R_d'] = compute_dynamic_efficiency(
                        file_path=params['machine_data_file'],
                        target_date=now_dt, # Uses datetime
                        R_block=state['R_block'],
                        C_elec=params['C_elec'],
                        C_elec_0=params.get('C_elec_0', params['C_elec']),
                        P_BTC=state['P_USD'],
                        TX_FEE_btc=state['aggregate_daily_fees'],
                        daily_difficulties=daily_difficulties,
                        daily_block_times=daily_block_times,
                        daily_block_subsidies=daily_subsidies,
                        total_operating=params['total_operating'],
                        historical_cutoff=params['last_hist_time'],
                        modern_era=modern_era
                    )
                
                state['R_t'] = economics.calc_expected_revenue(state['R_block'], state['TX_fee'], state['target'])
                state['R_pe'] = economics.calc_price_energy_adjusted_revenue(state['R_t'], state['P_USD'], state['efficiency'])
            else:
                state['R_t'] = economics.calc_expected_revenue(state['R_block'], state['TX_fee'], state['target'])
                state['R_pe'] = np.nan

        else: 
            # FORECAST MODE
            # The first forecast-mode block must be the one modules/boundaries.py assigns
            # to the cutoff (its boundary block + 1); run_simulation sets the expected
            # value when the cutoff lies inside the block data.
            if params.get('_first_forecast_block') is None:
                params['_first_forecast_block'] = block_height
                expected = params.get('_expected_first_forecast_block')
                if expected is not None and block_height != expected:
                    raise RuntimeError(
                        f"the economic model took over at block {block_height}, but the UTC "
                        f"boundary rule puts the first forecast block of cutoff "
                        f"{params['historical_cutoff']} at {expected}")
            fix_flag = params.get('fix_efficiency', False)
            eff_cutoff = params['last_hist_time']
            
            if fix_flag and now_dt > eff_cutoff:
                if params['fixed_efficiency'] is None:
                    params['fixed_efficiency'] = state['efficiency']
                state['efficiency'] = params['fixed_efficiency']
            else:
                state['efficiency'], state['R_d'] = compute_dynamic_efficiency(
                    file_path=params['machine_data_file'],
                    target_date=now_dt,
                    R_block=state['R_block'],
                    C_elec=params['C_elec'],
                    C_elec_0=params.get('C_elec_0', params['C_elec']),
                    P_BTC=state['P_USD'],
                    TX_FEE_btc=state['aggregate_daily_fees'],
                    daily_difficulties=daily_difficulties,
                    daily_block_times=daily_block_times,
                    daily_block_subsidies=daily_subsidies,
                    total_operating=params['total_operating'],
                    historical_cutoff=params['last_hist_time'],
                    modern_era=modern_era
                )
            
            state['R_t'] = economics.calc_expected_revenue(state['R_block'], state['TX_fee'], state['target'])
            state['R_pe'] = economics.calc_price_energy_adjusted_revenue(state['R_t'], state['P_USD'], state['efficiency'])
            
            mean_bt = economics.economic_to_block_time(
                R_pe=state['R_pe'],
                base_time=params.get('expected_block_time', 600),
                C_elec_1=params['C_elec'],
                S_1=S_eff,
                C_elec_0=params['C_elec_0'],
                S_0=params['S_0'],
                ts=now_dt,
                modern_era=modern_era
            )

            if cal_mode:
                state['T_block'] = mean_bt
            else:              
                state['T_block'] = mean_bt
                # state['T_block'] = network.calc_stochastic_block_time(mean_bt) # Option to re-enable stochastic

            state['mean_bt'] = mean_bt
            
            # Advance simulation time
            _advance_clock(state)

        return state

    except Exception as e:
        msg = (
            f"Error in simulate_step at block {block_height}: {e}\n"
            + traceback.format_exc()
        )
        debug.log_state(state, msg)
        raise

def run_simulation(initial_state: dict, params: dict, num_steps: int, initial_block_height: int = 0) -> list:
    state = initial_state.copy()
    
    # Ensure sim_timestamp_s is present
    if 'sim_timestamp_s' not in state:
        # Fallback if not initialized: convert existing datetime (UTC-naive) to seconds
        ts = state.get('sim_timestamp')
        if ts:
             state['sim_timestamp_s'] = utc_seconds(ts)
        else:
             state['sim_timestamp_s'] = GENESIS_TIME_S
    state['sim_timestamp_s'] = int(state['sim_timestamp_s'])
    # The integer seconds are the source of truth; the datetime and the exact float
    # clock (see _advance_clock) are derived from them.
    state['sim_timestamp'] = pd.to_datetime(state['sim_timestamp_s'], unit="s", utc=True).tz_convert(None)
    state['sim_clock_s'] = float(state['sim_timestamp_s'])

    bp = params['block_paces']
    # A run that starts at a block in the data must start at that block's real UTC time
    # (block 0: the genesis time, 1231006505).
    if initial_block_height in bp.index:
        real_ts = int(bp.loc[initial_block_height, 'Block_Time_Seconds'])
        if state['sim_timestamp_s'] != real_ts:
            raise ValueError(
                f"initial state is stamped {state['sim_timestamp_s']} but block "
                f"{initial_block_height} is stamped {real_ts} (UTC) in the block data")

    state.setdefault('last_retarget_ts', state['sim_timestamp'])

    # Period boundaries by block height (modules/boundaries.py). The era split comes
    # from the UTC block times unless the caller passes it.
    if 'last_early_era_height' not in params:
        params['last_early_era_height'] = boundary_block(bp, ERA_SPLIT_UTC)
    # Expected first forecast-mode block: the cutoff's boundary block + 1, when the run
    # starts in historical mode at or before that boundary block and the cutoff lies
    # inside the block data. simulate_step checks it when the switch happens.
    params['_first_forecast_block'] = None
    params['_expected_first_forecast_block'] = None
    cutoff = params.get('historical_cutoff')
    if isinstance(cutoff, (datetime, pd.Timestamp)) and len(bp) > 0:
        try:
            b_cut = boundary_block(bp, cutoff)
        except ValueError:
            b_cut = None  # no block in the data is stamped after the cutoff
        if (b_cut is not None and initial_block_height <= b_cut
                and state['sim_timestamp_s'] <= utc_seconds(cutoff)):
            params['_expected_first_forecast_block'] = b_cut + 1
    history = [copy.copy(state)]
    block_height = initial_block_height

    # Per-run efficiency state (moving averages, last day, profitable-set mask) is
    # module-level; clear it so a second run in this process starts fresh.
    reset_run_state()

    for step in range(num_steps):
        block_height += 1

        # --- Difficulty retargeting ---
        if (block_height % params['period_blocks'] == 0) and (block_height >= params['period_blocks']):
            try:
                # Retrieve timestamps strictly as seconds from history
                # History index 0 matches block 0 (genesis).
                # To retarget at block 2016, we look at history for block 2015 (last of prev period)
                # and block 0 (first of prev period).
                
                # history[-1] is the state *after* block_height-1 was mined.
                last_ts_s = history[-1]["sim_timestamp_s"]
                
                # We need the timestamp of the block *before* the 2016 block window started.
                # Actually, standard adjustment uses: timestamp of block (H-1) - timestamp of block (H-2017)?
                # No, Bitcoin uses: (Time of block 2015) - (Time of block 0) for the first retarget at 2016.
                # history[-1] is block_height - 1.
                # history[-2016] is block_height - 2016.
                
                first_ts_s = int(history[-params["period_blocks"]]["sim_timestamp_s"])
                last_ts_s  = int(history[-1]["sim_timestamp_s"])
                
                new_target = network.update_target_from_seconds(prev_target=state["target"], first_ts_s=first_ts_s, last_ts_s=last_ts_s)
                
                state['target'] = new_target
                state['D'] = network.update_difficulty(new_target, params['default_target'])
                state['difficulty_adjusted'] = True
            except Exception as e:
                error_message = (
                    f"Error during difficulty retargeting at block {block_height}: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                debug.log_state(state, error_message)
                raise
        else:
            state['difficulty_adjusted'] = False

        state['block_height'] = block_height
        try:
            state = simulate_step(state, params, block_height, history)
        except Exception as e:
            debug.log_state(state, f"Simulation halted at block {block_height} due to error: {e}")
            raise

        history.append(copy.copy(state))

        if block_height % 10000 == 0:
            print(f"\rCurrent block height: {block_height}", end="", flush=True)

    print()
    return history