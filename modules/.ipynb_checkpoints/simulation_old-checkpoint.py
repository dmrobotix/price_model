import copy
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
from modules import network, economics, energy, debug
from modules.efficiency_cbeci import compute_dynamic_efficiency
from modules.price import get_price_for_date, forecast_price
from modules.economics import apply_hashrate_shock
from config import EXPECTED_BLOCK_TIME, S as S_CONFIG, CALIBRATION_MODE, TX_FEE_PCT

def simulate_step(state: dict, params: dict, block_height: int, history: list) -> dict:
    """
    Perform one simulation step by updating the simulation state.

    This function:
      - Updates block subsidy based on halving intervals.
      - Retrieves the current market price.
      - Computes dynamic efficiency.
      - Updates revenue and calculates the block time (historical or economic).
      - Advances the simulation timestamp.
    """
    from datetime import timedelta
    import traceback

    for key in ('target', 'D', 'H', 'sim_timestamp'):
        if key not in state:
            raise ValueError(f"Missing required state key: '{key}'.")

    try:
        # --- Halving rule ---
        halvings = block_height // params['halving_interval']
        state['R_block'] = params['initial_block_subsidy'] / (2**halvings)

        # --- Market price ---
        # NOTE: 'now' is the timestamp *before* the current block is mined
        now = state['sim_timestamp']
        if now > params['last_hist_time']:
            # Forecast region: use pre-fit forecaster
            state['P_USD'] = params['price_forecaster'](now)
        else:
            # Historical region: use fast streaming lookup
            # state['P_USD'] = get_price_for_date(params['market_prices'], now)
            state['P_USD'] = params['price_lookup'](now)

        # --- Determine block time ---
        enabled = params.get('shock_enabled', False)
        cal_mode = params.get('calibration_mode', CALIBRATION_MODE)
        S_eff = params.get('S', S_CONFIG) if cal_mode else S_CONFIG
        cutoff = params['historical_cutoff']
        bp = params['block_paces']

        # --- Gather daily data for CBECI (last complete calendar day, optimized) ---
        if not cal_mode or (cal_mode and now >= cutoff):
            # The 'history' list contains all blocks *up to* this new one.
            current_day = now.date()
            n_hist = len(history)

            # Rolling start index tracks the first block of the *most recently aggregated day*.
            start_idx = params.get('daily_window_start_idx', 0)
            
            # The day whose data is currently cached and being used in efficiency calculation.
            aggregated_day = params.get('daily_window_day', None)
            
            # The day we *want* to aggregate is the one that just completed: aggregated_day.
            # We only perform the full aggregation once per calendar day when 'current_day' changes.
            day_to_aggregate = None
            if aggregated_day is not None and current_day > aggregated_day:
                # If we've advanced to a new calendar day, the previous day (aggregated_day) is now complete.
                day_to_aggregate = aggregated_day
            
            # --- Aggregation logic runs once when a day is complete ---
            if day_to_aggregate is not None:
                
                # 1. Advance start_idx until we reach the first block of the day_to_aggregate.
                # (Rolling-window optimization: this skips blocks from days before day_to_aggregate.)
                while (
                    start_idx < n_hist
                    and history[start_idx]['sim_timestamp'].date() < day_to_aggregate
                ):
                    start_idx += 1
                
                # 2. Find the end of the day_to_aggregate block slice.
                end_idx = start_idx
                while (
                    end_idx < n_hist
                    and history[end_idx]['sim_timestamp'].date() == day_to_aggregate
                ):
                    end_idx += 1
                
                # The full day's history for the *previous* day (now complete)
                same_day_history = history[start_idx:end_idx]

                if not same_day_history:
                    # Fallback to current state if a full day's worth of history is missing
                    daily_difficulties = np.array([state['D']])
                    last_entry = history[-1] if n_hist > 0 else {}
                    last_T_block = last_entry.get('T_block', EXPECTED_BLOCK_TIME)
                    last_TX_fee = last_entry.get('TX_fee', 0.0)
                    daily_block_times = np.array([last_T_block])
                    daily_tx_fees = np.array([last_TX_fee])
                else:
                    daily_difficulties = np.array([h['D'] for h in same_day_history])
                    daily_block_times = np.array(
                        [h.get('T_block', EXPECTED_BLOCK_TIME) for h in same_day_history]
                    )
                    daily_tx_fees = np.array(
                        [h.get('TX_fee', 0.0) for h in same_day_history]
                    )
                
                # --- Aggregate daily transaction fees (BTC) ---
                aggregate_daily_fees = float(np.sum(daily_tx_fees))
                
                # --- Cache the results for the full day, to be used for every block of 'current_day' ---
                params['daily_difficulties_cached'] = daily_difficulties
                params['daily_block_times_cached'] = daily_block_times
                params['aggregate_daily_fees_cached'] = aggregate_daily_fees
                
                # --- Update the rolling window state for the *next* day's aggregation ---
                # start_idx for the next full day aggregation should start at 'end_idx' of the current one.
                params['daily_window_start_idx'] = end_idx 
                
                # Update the day being tracked to the *new* day we just entered
                params['daily_window_day'] = current_day 
            
            # --- Use the cached (full day) results for the current block ---
            # If no aggregation has happened yet (first day), use defaults.
            daily_difficulties = params.get('daily_difficulties_cached', np.array([state['D']]))
            daily_block_times = params.get('daily_block_times_cached', np.array([state.get('T_block', EXPECTED_BLOCK_TIME)]))
            aggregate_daily_fees = params.get('aggregate_daily_fees_cached', 0.0)
            
            state['aggregate_daily_fees'] = aggregate_daily_fees
            
            # Ensure 'daily_window_day' is set for the very first block to enable the 'new day' check.
            if params.get('daily_window_day') is None:
                params['daily_window_day'] = current_day 

        elif cal_mode and now < cutoff:
            # Calibration mode before cutoff uses historical block times directly
            # and does NOT need daily CBECI data. Just set a dummy aggregate_daily_fees.
            state['aggregate_daily_fees'] = 0.0


        # --- Historical vs Forecast logic ---
        if now <= cutoff: # historical logic
            # Use historical block interval and transaction fee
            state['T_block'] = float(bp.loc[block_height, 'Inter_Block_Interval_Seconds'])
            state['TX_fee'] = float(bp.loc[block_height, 'Total_Fees_BTC'])
            #hist_ts = bp.loc[block_height, 'Time'] # The historical timestamp *after* the block
            hist_ts_s = int(bp.loc[block_height, "Block_Time_Seconds"])
            state["sim_timestamp_s"] = hist_ts_s
            state["sim_timestamp"] = pd.to_datetime(hist_ts_s, unit="s", utc=True).tz_convert(None)


            if not cal_mode: 
                # --- Efficiency ---
                fix_flag = params.get('fix_efficiency', False)
                eff_cutoff = params['last_hist_time']
                
                if fix_flag and now > eff_cutoff:
                    if params['fixed_efficiency'] is None:
                        params['fixed_efficiency'] = state['efficiency']
                    state['efficiency'] = params['fixed_efficiency']
                else:
                    # This is the new, corrected function call
                    state['efficiency'], state['R_d'] = compute_dynamic_efficiency(
                        file_path=params['machine_data_file'],
                        target_date=now,
                        R_block=state['R_block'],
                        C_elec=params['C_elec'],
                        C_elec_0=params.get('C_elec_0', params['C_elec']), # hobby era (fallback to C_elec)
                        P_BTC=state['P_USD'],
                        TX_FEE_btc=state['aggregate_daily_fees'],
                        daily_difficulties=daily_difficulties,    # Added daily data
                        daily_block_times=daily_block_times,      # Added daily data
                        total_operating=params['total_operating'],
                        historical_cutoff=params['last_hist_time'] # Passed this from params
                    )
                
                # --- Revenue metrics ---
                state['R_t'] = economics.calc_expected_revenue(state['R_block'], state['TX_fee'], state['target'])
                state['R_pe'] = economics.calc_price_energy_adjusted_revenue(state['R_t'], state['P_USD'], state['efficiency'])
            else: # calibrates S in the economic model
                # do not compute efficiency or R_pe for historical if cal_mode is TRUE
                # --- Revenue metrics ---
                state['R_t'] = economics.calc_expected_revenue(state['R_block'], state['TX_fee'], state['target'])
                state['R_pe'] = np.nan
                
            
            # --- ADVANCE SIMULATION TIME (FIXED LOGICAL ISSUE) ---
            # Use the historical timestamp if it's before or at the cutoff, 
            # or the calculated T_block otherwise for the transition.
            # if hist_ts <= cutoff:
            #     state['sim_timestamp'] = hist_ts
            # else:
            #     state['sim_timestamp'] += timedelta(seconds=state['T_block'])
            hist_ts_s = int(bp.loc[block_height, "Block_Time_Seconds"])
            state["sim_timestamp_s"] = hist_ts_s
            state["sim_timestamp"] = pd.to_datetime(hist_ts_s, unit="s", utc=True).tz_convert(None)


        else: # forecasting logic
            # --- Efficiency ---
            state['TX_fee'] = TX_FEE_PCT*state['R_block']
            fix_flag = params.get('fix_efficiency', False)
            eff_cutoff = params['last_hist_time']
            
            if fix_flag and now > eff_cutoff:
                if params['fixed_efficiency'] is None:
                    params['fixed_efficiency'] = state['efficiency']
                state['efficiency'] = params['fixed_efficiency']
            else:
                # This is the new, corrected function call
                state['efficiency'], state['R_d'] = compute_dynamic_efficiency(
                    file_path=params['machine_data_file'],
                    target_date=now,
                    R_block=state['R_block'],
                    C_elec=params['C_elec'],
                    C_elec_0=params.get('C_elec_0', params['C_elec']), # hobby era (fallback to C_elec)
                    P_BTC=state['P_USD'],
                    TX_FEE_btc=state['aggregate_daily_fees'],
                    daily_difficulties=daily_difficulties,    # Added daily data
                    daily_block_times=daily_block_times,      # Added daily data
                    total_operating=params['total_operating'],
                    historical_cutoff=params['last_hist_time'] # Passed this from params
                )
            
            # --- Revenue metrics ---
            state['R_t'] = economics.calc_expected_revenue(state['R_block'], state['TX_fee'], state['target'])
            state['R_pe'] = economics.calc_price_energy_adjusted_revenue(state['R_t'], state['P_USD'], state['efficiency'])
            
            # Economic model block time
            mean_bt = economics.economic_to_block_time(
                R_pe=state['R_pe'],
                base_time=params.get('expected_block_time', 600),
                C_elec_1=params['C_elec'],
                S_1=S_eff,
                C_elec_0=params['C_elec_0'],
                S_0=params['S_0'],
                ts=now
            )

            if cal_mode: # calibrates S in the economic model
                state['T_block'] = mean_bt
            else:              
                #state['T_block'] = network.calc_stochastic_block_time(mean_bt)
                state['T_block'] = mean_bt

            # Advance simulation time
            state['sim_timestamp'] += timedelta(seconds=state['T_block'])

            # store mean_bt
            state['mean_bt'] = mean_bt

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
    state.setdefault('last_retarget_ts', state['sim_timestamp'])
    history = [copy.copy(state)]
    block_height = initial_block_height

    for step in range(num_steps):
        block_height += 1

        # --- Difficulty retargeting BEFORE simulating block_height (Core-style) ---
        if (block_height % params['period_blocks'] == 0) and (block_height >= params['period_blocks']):
            try:
                # history has blocks 0 .. block_height-1 at this point
                # Last block of the previous period: H-1
                # last_ts  = history[-1]['sim_timestamp']
                # # First block of that previous period: H-2016
                # first_ts = history[-params['period_blocks']]['sim_timestamp']

                # new_target = network.update_target(state['target'], first_ts, last_ts)
                last_ts_s  = int(history[-1]["sim_timestamp_s"])
                first_ts_s = int(history[-params["period_blocks"]]["sim_timestamp_s"])
                
                new_target = network.update_target_from_seconds(prev_target=state["target"], first_ts_s=first_ts_s,last_ts_s=last_ts_s)
                
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

        # --- Now simulate the block with the current target/D ---
        state['block_height'] = block_height
        try:
            state = simulate_step(state, params, block_height, history)
        except Exception as e:
            debug.log_state(state, f"Simulation halted at block {block_height} due to error: {e}")
            raise

        history.append(copy.copy(state))

        if block_height % 100000 == 0:
            print(f"\rCurrent block height: {block_height}", end="", flush=True)

    print()
    return history