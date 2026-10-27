from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid, aluSimplexGrid

domain = cartesianDomain([0,0],[1,1],[1,1])
aluView = aluSimplexGrid(domain)
hGrid = aluView.hierarchicalGrid
hGrid.globalRefine(5)
for level in range(hGrid.maxLevel):
    print("level:",level, "number of elements:",hGrid.levelView(level).size(0))
aluView = aluConformGrid(domain)
hGrid = aluView.hierarchicalGrid
hGrid.globalRefine(10)
for level in range(hGrid.maxLevel):
    print("level:",level, "number of elements:",hGrid.levelView(level).size(0))