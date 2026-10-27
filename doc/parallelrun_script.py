import subprocess
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Code, display

try:
    import mpi4py
except ImportError:
    print("Example requires an `mpi` installation - skipping")
    sys.exit(0)
oversub = ["--map-by", ":OVERSUBSCRIBE"] # allow oversubscription for this simple test
nprocs = 3
print("==========================================")
print(["mpiexec", *oversub, "-n", str(nprocs), sys.executable, "crystal.py"],flush=True)
print("==========================================")
subprocess.run(
    ["mpiexec", *oversub, "-n", str(nprocs), sys.executable, "crystal.py"],
    check=True,
)
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
nprocs = 2
print("==========================================")
print(["mpiexec", *oversub, "-n", str(nprocs), sys.executable, "wave_mpi.py"],flush=True)
print("==========================================")
subprocess.run(
    ["mpiexec", *oversub, "-n", str(nprocs), sys.executable, "wave_mpi.py"],
    check=True,
)
fig, axes = plt.subplots(1, nprocs, figsize=(5*nprocs, 4))
for rank, ax in enumerate(axes):
    image = mpimg.imread(f"wave_mpi{rank}.png")
    ax.imshow(image)
    ax.set_title(f"Rank {rank}")
    ax.axis("off")
fig.tight_layout()
display(Code(filename="wave_mpi.py", language="python"))