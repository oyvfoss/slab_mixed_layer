### SLAB_MIXED_LAYER ###

Evaluating the mixed layer response to wind forcing using the 1-D slab mixed
layer model of Pollard and Millard (1970).

For now: Solving with a simple forward scheme - will transfer to a more
optimized method (probably scipy.integrate.solve_ivp).

---

STATUS 26.06 2020: 

Setup project. Lacking documentation, validation, and test scripts, and
the numerical integration is somewhat clunky.

---

References
==========

Pollard, R. T., & Millard Jr, R. C. (1970, August). Comparison between observed
and simulated wind-generated inertial oscillations. In Deep Sea Research and 
Oceanographic Abstracts (Vol. 17, No. 4, pp. 813-821). Elsevier.