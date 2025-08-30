# ExoRVFisher (*working name*) 
Scripts to calculate and visualize the Fisher information content for radial velocity measurements of exoplanet orbits.
Adapted from the code used in [Gupta & Bedell et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024AJ....168...29G/abstract), with added functionality to handle eccentric orbits.

# Installation

Clone the repository (`git clone https://github.com/afgupta/ExoRVFisher`) and install the following dependencies:
* `pandas`
* `scipy`
* `matplotlib` (if using plotting features)
* `jax`<0.6 (*must* be <0.6 for compatibility with `exoplanet-core`)
* `exoplanet-core` (install as `"python3 -m pip install "exoplanet-core[jax]"`)

# Code Description

The code presently consists of two main scripts: `run_fisher_grid.py` and `run_fisher_progression.py`, along with several helper scripts.
Both of these are run via the command line and accept a number of arguments and optional flags. The primary inputs are (1) a csv containing
the details of the set of observations to be analyzed, (2) a configuration file (see the `/config/` folder for examples), (3) a csv with
input stellar parameters, and (4) a csv with orbital parameters for the planet(s) of interest.

`run_fisher_grid.py` calculates the expected Fisher information over a grid of orbital parameters, with bounds and resolution specified in
the input configuration file. It can optionally produce a plot similar to Figure 6 of [Gupta & Bedell et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024AJ....168...29G/abstract), e.g.: 

<img src=https://github.com/afgupta/ExoRVFisher/blob/main/Example_results/Example_TestStar2_sensitivity_02B.png width=60% height=60%>

`run_fisher_progression.py` calculates the expected Fisher information as a function of the number of input observations for a single set of
orbital parameters. It can optionally produce a plot similar to Figure 7 of [Gupta & Bedell et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024AJ....168...29G/abstract), e.g.:

<img src=https://github.com/afgupta/ExoRVFisher/blob/main/Example_results/Example_fisher_progress_TestPlanet1_03A.png width=60% height=60%>

# Examples

Some example results are provided in `/Example_results/`; these were created using the sample observation files (in `/survey_files`),
configuration files (in `/config`) and stellar and planetary parameter files (in `/params/`). The results can be reproduced with the
following command line inputs:

**01A**: `python3 -m run_fisher_grid Example TestStar1 01A -i example_observations_1.csv -g test01A.config -s -p`

**01B**: `python3 -m run_fisher_grid Example TestStar1 01B -i example_observations_1.csv -g test01B.config -s -p`

**02A**: `python3 -m run_fisher_grid Example TestStar2 02A -i example_observations_2.csv -g test02A.config -s -p`

**02B**: `python3 -m run_fisher_grid Example TestStar2 02B -i example_observations_2.csv -g test02B.config -s -p`

**03A**: `python3 -m run_fisher_progression Example TestPlanet1 03A -i example_observations_3.csv -g test03A.config -e -s -p`
