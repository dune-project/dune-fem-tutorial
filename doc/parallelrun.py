# %% [markdown]
# .. index:: Parallelization; MPI
#
# # Distributed Parallel Simulations
#
# We will run two of the examples already described in parallel to
# demonstrate how little to no change is required to do so.

# %%
import subprocess
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    import mpi4py
except ImportError:
    print("Example requires an `mpi` installation - skipping")
    sys.exit(0)

oversub = ["--map-by", ":OVERSUBSCRIBE"] # allow oversubscription for this simple test

# %% [markdown]
# .. index:: Adaptation; Phase Field; MPI
#
# ## Parallel Crystal Growth Model
#
# We now run the [crystal growth phase field example](crystal_nb.ipynb):

# %%
nprocs = 3
print("==========================================")
print(["mpiexec", *oversub, "-n", str(nprocs), sys.executable, "crystal.py"],flush=True)
print("==========================================")
subprocess.run(
    ["mpiexec", *oversub, "-n", str(nprocs), sys.executable, "crystal.py"],
    check=True,
)

# %% [markdown]
# Each process produced a plot containing only part of the grid
# at five different point in the simulation. Note how the call
# `fem.gridAdapt(marker, u_h)` within the time loop does not only refine
# and coarsen the grid but also leads to occasional repartitioning of
# the grid to balance the load between the three processors.
# We first use pvpython if available to generate figures showing the
# combined grid and solution with the partitioning. Then we show the
# individual domains - those figures were generated during the above call
# to `crystal.py`:

# %%
print("============================================")
try:
    for i in range(6):
        print(f"pvpython call {i}")
        subprocess.run(['pvpython','pvcrystal.py', str(nprocs), str(i)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
        )
    figGenerated = True
except: # probably no pvpython available
    figGenerated = False
print("============================================")
for i in range(6):
    fig, axes = plt.subplots(1, nprocs+1, figsize=(5*(nprocs+1), 4),
                             gridspec_kw={"width_ratios": [3] + [1] * nprocs})
    for rank, ax in enumerate(axes[1:]):
        image = mpimg.imread(f"crystal{rank}{nprocs}_{i}.png")
        ax.imshow(image)
        ax.set_title(f"Rank {rank}")
        ax.axis("off")
    if figGenerated:
        image = mpimg.imread(f"crystal{nprocs}_{i}.png")
        axes[0].imshow(image)
        axes[0].set_title(f"Full")
        axes[0].axis("off")
    fig.tight_layout()

# %% [markdown]
# .. index:: Equations; Wave Equation; MPI
#
# ## Parallel Wave Equation
#
# We now run the [double slit domain wave example](wave_nb.ipynb)
# on two processes with a distributed grid. Here we need to make a slight
# adjustment to one part of the original code which is detailed below.
# Note that this change could have been included in the serial version
# without any issue. To keep the explanation simple there we decided
# to include the change in the separate script.

# %%
nprocs = 2
print("==========================================")
print(["mpiexec", *oversub, "-n", str(nprocs), sys.executable, "wave_mpi.py"],flush=True)
print("==========================================")
subprocess.run(
    ["mpiexec", *oversub, "-n", str(nprocs), sys.executable, "wave_mpi.py"],
    check=True,
)

# %% [markdown]
# Again each process produced a plot containing its part of the grid

# %%
fig, axes = plt.subplots(1, nprocs, figsize=(5*nprocs, 4))
for rank, ax in enumerate(axes):
    image = mpimg.imread(f"wave_mpi{rank}.png")
    ax.imshow(image)
    ax.set_title(f"Rank {rank}")
    ax.axis("off")
fig.tight_layout()

# %% [markdown]
# Here is the code being run - note how almost no additional code is needed
# compared to the [serial version](wave_nb.ipynb) with the exception
# of the setup for the inverse lumped mass matrix

# %%
from IPython.display import Code, display
display(Code(filename="wave_mpi.py", language="python"))
