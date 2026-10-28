from dune.fem import threading
from functools import partial
from dune.common import comm
from dune.alugrid import aluCubeGrid
from dune.grid import cartesianDomain
from dune.fem.function import partitionFunction

print("Using",threading.use,"threads")
threading.use = 4 # use 4 threads
print("Using",threading.use,"threads")
print("Maximum number of threads available:",threading.max)
threading.useMax()
print("Using",threading.use,"threads")
# print can be used as before but will only produce output on rank 0
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None
domain = cartesianDomain([0,0], [1,1], [40,40], overlap=1)
gridView = aluCubeGrid( domain, lbMethod=9 )
# distribute work load, no user data is adjusted
gridView.hierarchicalGrid.loadBalance()
# print possible load balancing methods
help( aluCubeGrid )
vtk = gridView.sequencedVTK("data", celldata=[partitionFunction(gridView)])
vtk()