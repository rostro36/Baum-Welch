# Plotting

## Scripts
- report_N_plotting.py: creates a performance plot for gcc , a performance plot for the comparison between gcc and icc and roofline model plot with gcc for stb, cop, reo and vec.
- report_bla_umdhmm.py: creates a cycle/iteration plot for all previous mentioned versions and the BLAS version as well as the third party library umdhmm
- report_machines.py: creates a performance plot for two different machines for stb, cop, reo ( can also be used for vec if both machines support fma instructions)
- report_plotting.py: creates performance plot for the impact of one parameter for stb, cop, reo and vec

## Usage
python3 $script_name.py
