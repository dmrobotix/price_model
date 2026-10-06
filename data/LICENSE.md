# License of `data/cbeci_machines_090325.csv`

This notice covers one file, `cbeci_machines_090325.csv`, the machine table that the
model reads by default (`MACHINE_DATA_FILE` in `config.py`). It is the only data file
published in this folder. The other input files are not in this repository; see
`README.md`.

The file is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International licence (CC BY-NC-SA 4.0),
https://creativecommons.org/licenses/by-nc-sa/4.0/.

Source: the CBECI SHA-256 Mining Equipment List of the Cambridge Bitcoin Electricity
Consumption Index (CBECI), Cambridge Centre for Alternative Finance,
https://ccaf.io/cbnsi/cbeci, licensed under CC BY-NC-SA 4.0. The file is the copy of
that list used for the paper. The repository's `README.md` records that it was
downloaded from CBECI's website on 9 March 2025; the date in this notice comes from
there. The `090325` in the file name is that date, written as day, month and year.

Columns: the file keeps eight columns of CBECI's list (`Miner_name`, `Type`,
`Date of release`, `UNIX_date_of_release`, `Hashing Power (TH/s)`, `Power (W)`,
`Efficiency (J/Gh)`, `Weight in kg`). CBECI's list also has the columns
`Efficiency: suggested alternative(s)`, `Included in  version` and `Additional comments`,
which the file does not contain.

The file was compared with CBECI's list as fetched on 4 October 2026. The results below
come from `validation/results/machine_list_check/summary.txt` (section 1):

- The file has 166 machines. All 166 are in CBECI's list under the same name. CBECI's
  list has 15 further machines (versions 1.6.5 to 1.8.0, released December 2024 to
  April 2026) that the file does not contain.
- The release date is identical for all 166 machines.
- The hashing power is identical for 144 of the 166 machines, and the power for 163.
- The efficiency differs from CBECI's by at most 0.155% (Canaan Avalon A1466). It
  equals CBECI's value rounded to four decimals of J/Gh for 164 of the 166 machines.
- The file's header writes `Hashing Power (TH/s)`. CBECI's writes `Hashing power (Th/s)`.

`UNIX_date_of_release` is written with thousands separators, for example
`"1,230,768,000"`. CBECI's own CSV export of the list writes it the same way.

The origin of the differences listed above (the missing machines, the hashing power,
power and efficiency values, and the header) is not established. The repository cannot show
whether they are revisions CBECI made to its list after March 2025, or edits made to
the file after it was downloaded.

CBECI does not endorse this repository or the model. The code in this repository is
licensed separately under the PolyForm Noncommercial License 1.0.0 (see `LICENSE.md` at
the repository root).
