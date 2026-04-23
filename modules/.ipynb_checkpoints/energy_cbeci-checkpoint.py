# modules/energy.py

"""
Energy Module

This module contains functions for computing energy-related quantities in the mining simulation.
Functions include:
  - calc_power_demand: Converts network hashrate into power demand (in Watts) given machine efficiency.
  - calc_energy_consumption: Calculates energy consumption over a given period (in Watt-hours).
  - compute_water_consumption: Calculates the change in water consumption (in liters per day) based on energy usage.
"""

def calc_power_demand(hashrate: float, efficiency: float) -> float:
    """
    Calculate the power demand given the network hashrate and machine efficiency.
    
    Args:
        hashrate (float): The network hashrate (in hashes per second).
        efficiency (float): The efficiency of the mining hardware (in Joules per hash).
        
    Returns:
        float: The power demand in Watts (since 1 Watt = 1 Joule/second).
    """
    return hashrate * efficiency

def calc_energy_consumption(power: float, hours: float) -> float:
    """
    Calculate the energy consumption over a given period.
    
    Args:
        power (float): The power demand in Watts.
        hours (float): The duration in hours.
        
    Returns:
        float: The energy consumed in Watt-hours (Wh).
    """
    return power * hours

def compute_water_consumption(E_HPC: float, W_air: float, W_water_mining: float, W_HPC: float) -> float:
    """
    Compute the change in water consumption (Delta W) based on energy usage.
    
    Args:
        E_HPC (float): Energy associated with capacity that exited (in MWh/day).
        W_air (float): Water usage factor for dry-air cooling (liters per MWh).
        W_water_mining (float): Water usage factor for water-cooled mining (liters per MWh).
        W_HPC (float): Water usage factor for HPC/AI applications (liters per MWh).
        
    Returns:
        float: The change in water consumption (liters per day).
    """
    # For mining, assume an effective water usage is a weighted combination:
    # 80% from dry-air and 20% from water-cooling.
    W_mining = 0.8 * W_air + 0.2 * W_water_mining
    W_mining_lost = E_HPC * W_mining  # Water consumption if capacity was mined
    W_HPC_usage = E_HPC * W_HPC       # Water consumption if capacity is used for HPC/AI
    return W_HPC_usage - W_mining_lost
