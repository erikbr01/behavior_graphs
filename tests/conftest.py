"""Cap the numeric libraries' thread pools before anything imports them.

`-n auto` starts one worker per core and each worker's BLAS/OpenMP runtime then
wants a thread per core on top, so 32 workers ask for ~1024 threads on 32 cores.
Measured on this box, back to back, default tier:

    -n 8,  threads default   30 s, 78 s
    -n 8,  threads 1         21 s, 26 s
    -n 32, threads default   50 s, 83 s
    -n 32, threads 1         27 s, 26 s

The cap is worth 2-3x and it is what makes the worker count stop mattering --
uncapped, the suite has a sharp knee above -n 8; capped, -n 8 and -n 32 land
within a second of each other. Every test here is small enough that one thread
per worker loses nothing: the parallelism that matters is across workers.

`setdefault`, so `OMP_NUM_THREADS=8 pytest ...` still wins if you want it.

This file must stay import-light and must not import numpy, torch or sklearn --
the variables are only read when those libraries initialise, so setting them
after an import somewhere else has no effect.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# Headless, for the same reason and in the same import-light way: matplotlib
# reads this when it initialises, so an env var here beats importing it to call
# `use()`. Left to itself on a Mac it picks the MacOSX backend, and the page
# tests draw their figures off the main thread -- "Cannot create a GUI
# FigureManager outside the main thread". That only bites when something that
# imports matplotlib eagerly (arviz, by way of bambi) happens to run first, so
# the suite passed in file order and failed the moment two files were named on
# the command line the other way round.
os.environ.setdefault("MPLBACKEND", "Agg")
