# Miner Model

This repository contains the Python code used to run the miner economics forecasting and simulation model described in the paper. The model is run from `main.py`, with key settings and input file paths defined in `config.py`. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Requirements

Install the required Python packages first:

```bash
pip install -r REQUIREMENTS.txt
````

The required packages are NumPy, pandas, matplotlib, Plotly, SciPy, and scikit-learn. 

## Project files

The main files needed to run the model are:

* `main.py` — entry point for the simulation. 
* `config.py` — global settings, forecast assumptions, and input data paths. 
* `data_processing.py` — loads block pace and fee data. 
* `price.py` — loads historical market prices and builds the price forecast function. 
* `simulation.py` — runs the block-by-block simulation. 
* `efficiency_cbeci.py` — computes the dynamic mining hardware efficiency used by the model. 

## Input data

Before running the model, make sure the input data files referenced in `config.py` exist in the expected locations. In the current code, these include:

* historical block data
* transaction fee data
* historical price data
* mining machine data

These file paths are set through variables such as `BLOCK_PACE_DATA`, `TX_BLOCK_DATA`, `PRICE_DATA`, and `MACHINE_DATA_FILE` in `config.py`. 

## How to run

From the project root, run:

```bash
python main.py
```

This script:

1. loads the historical price and block data,
2. builds the price forecast function,
3. initializes the simulation state,
4. runs the block-by-block model,
5. saves the simulation output to a CSV file, and
6. generates plots. 

## Configuration

Most settings can be changed directly in `config.py`. Important options include:

* `FORECAST_MODEL` for the price forecast method,
* `FORECAST_TARGET_DATE` and `FORECAST_TARGET_PRICE` for forecast anchoring,
* `C_ELEC`, `C_ELEC_0`, `S`, and `S_0` for economic parameters,
* `EFFICIENCY_SCENARIO` for the hardware efficiency scenario,
* `PRICE_DATA`, `TX_BLOCK_DATA`, `BLOCK_PACE_DATA`, and `MACHINE_DATA_FILE` for input files. 

## Output

The model writes simulation results to a CSV file in the data directory and can also generate plots for hashrate, energy use, annual energy, price, and related diagnostics.  

## Notes

This code is intended to accompany the paper and reproduce the simulation workflow described there. Depending on your local folder structure, you may need to update the relative file paths in `config.py` before running the model. 


