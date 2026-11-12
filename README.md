# Emberger Q2 bioclimatic stages in northern Algeria, 1961-2025

Code for the manuscript "Bioclimatic boundaries on the move: a multi-product assessment of Emberger stage shifts in northern Algeria, 1961-2025" (Gana and Touaba, submitted to Theoretical and Applied Climatology).

The full deposit (scripts, result tables, rasters, figures; 97 MB) is archived on Zenodo: https://doi.org/10.5281/zenodo.24162524

## What this repository holds

Analysis scripts (`*.py`) and Windows launchers (`RUN_*.bat`). Earlier versions (`_v2`, `_v3`, ...) are kept for traceability; the highest version of each script is the one used for the manuscript, unless the manuscript says otherwise. Result tables, rasters and figures are on Zenodo, not here.

## Method in short

Q2 = 2000 P / (M^2 - m^2), with P annual precipitation (mm), M the mean daily maximum of the hottest month and m the mean daily minimum of the coldest month (kelvin). Stage limits are in `Algeria_Q2_Stage_Shift.py` (class limits 20, 40, 70, 100). Products: TerraClimate, CHIRPS, ERA5-Land, with GHCN-Daily stations for validation. The Sahara is excluded by a mask.

## Run order (approximate)

Scripts import one another, so keep them in one folder. Typical order: `Algeria_North_Station_Inventory.py`, `Algeria_Q2_Stage_Shift.py`, `Algeria_Grid_Q2_Shift.py`, `Algeria_Ensemble_Q2_Shift.py`, `Algeria_Windows_Transitions_v3.py`, `Algeria_Isoline_v5.py`, `Algeria_Robustness_v4b.py`, `Algeria_Mask_DEM_v7.py`, `Algeria_Wilaya_Zones.py`, `Algeria_Wilaya_Climagram_v3.py`, `Station_Validation.py`, `Station_Change_Validation_v2.py`, `Homogeneity_GHCN_v1.py`, `Algeria_Daget_Frameworks_v2.py`, `Algeria_Frameworks_C_v1.py`. Paths at the top of the scripts point to the authors' machine and must be edited.

## Not included

Raw climate products (download them from the providers), the GHCN-Daily cache, and the DEM and administrative boundaries used for the mask.

## Licence and citation

Code: MIT (see LICENSE). Data, maps and figures (on Zenodo): CC BY 4.0.

Cite as: Gana, M., Touaba, C. (2026). Data, code and figures for "Bioclimatic boundaries on the move: a multi-product assessment of Emberger stage shifts in northern Algeria, 1961-2025". Zenodo. https://doi.org/10.5281/zenodo.24162524
## Note on completeness

This repository holds the main analysis scripts. Some scripts named in the run order above (for example the Daget-framework, homogeneity and figure scripts) and all result files are in the Zenodo deposit, which is the complete, citable version: https://doi.org/10.5281/zenodo.24162524
