# Baum-Welch

Fast implementation of the [Baum-Welch algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) for the course ["Advanced Systems Lab"](https://acl.inf.ethz.ch/teaching/fastcode/2020/)(Spring 2020) at ETH Zürich.

## Index
- [code](/code/usage.md) contains the main versions of the algorithm and scripts to benchmark them
- [documents](/documents/) contains written documents like the counting of instructions or the final report.
- [experiments](/experiments/) shows some small scale tests that were run during the development of the algorithm and also the testing with a r-library
- [old versions](/old_versions/versions.md) shows the small increments and bigger tests between main versions
- [output_measures](/output_measures/) contains the output of the benchmark scripts with times and parsed cachegrind output.
- [plots](/plots/) displays all plots that were generated, the most important of those are also in the final report. Next to the plots are also the scripts to generate them.
- [valgrind](/valgrind/) has the raw [cachegrind](https://valgrind.org/docs/manual/cg-manual.html) output in them.

## Main versions
- std For the most standard implementations
- stb For the stable version
- cop For all basic C optimizations
- reo For the reordering (code motion) step
- bla For the BLAS version
- url For the unrolled version
- vec For the vectorized version

## Credits
This project was made by Team 35:
- Luca Blum (lblum)
- Jannik Gut (jgut)
- Jan Schilliger (schiljan)
