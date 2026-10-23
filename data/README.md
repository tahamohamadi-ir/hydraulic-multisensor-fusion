Source: UCI [Condition monitoring of hydraulic systems](https://archive.ics.uci.edu/dataset/447), DOI 10.24432/C5CW21, CC BY 4.0.

Default program input is `data/hydraulic_uci.zip` (either legacy `data.zip`, or new UCI outer zip containing `data.zip`). The program extracts only four needed files and `profile.txt` in memory. Alternatively: `python -m hydraulic.run --source data/raw/` where the folder may contain nested subfolders with raw sensor TXT files.

Data is not included in Git. Run `python scripts/get_data.py` or `python scripts/get_data.py --archive PATH_TO_LOCAL_UCI_ZIP`.
