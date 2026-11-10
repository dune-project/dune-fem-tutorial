import matplotlib.pyplot as plt
import numpy as np
import dune.ufl
from landlab import RasterModelGrid
from landlab.io import esri_ascii

if False:
    from landlab.components.overland_flow import LinearDiffusionOverlandFlowRouter as LDOF
else:
    from dune.femdg.landlab import DuneDiffOverlandFlowSolver as LDOF
# Process parameters
n = 0.1  # roughness coefficient, (s/m^(1/3))
uc = 1.0  # characteristic velocity scale (m/s)
R1 = 72.0 / (3600.0 * 1000.0)  # initial rainfall rate, m/s (converted from mm/hr)
R2 = 0.0 / (3600.0 * 1000.0)  # later rainfall rate, m/s (converted from mm/hr)
infilt_cap = 10.0 / (3600 * 1000.0)  # infiltration capacity, m/s (converted from mm/hr)
dx = 10
# Run-control parameters
heavy_rain_duration = 300.0  # duration of heavy rainfall, s
run_time = 1200.0  # duration of run, s
dt = 20.0  # time-step size, s
dem_filename = "hugo_site.asc"
# Derived parameters
num_steps = int(run_time / dt)
# set up arrays to hold discharge, time, and rain rate
time_since_storm_start = np.linspace(0.0, run_time, num_steps + 1)
discharge = np.zeros(num_steps + 1)
rain_rate = np.zeros(num_steps + 1)
rain_rate[:] = R1
rain_rate[time_since_storm_start >= heavy_rain_duration] = R2
# Read the DEM file as a grid with a 'topographic__elevation' field
with open(dem_filename) as fp:
    grid = esri_ascii.load(fp, name="topographic__elevation", at="node")
elev = grid.at_node["topographic__elevation"]
# Configure the boundaries: valid right-edge nodes will be open;
# all NODATA (= -9999) nodes will be closed.
grid.status_at_node[grid.nodes_at_right_edge] = grid.BC_NODE_IS_FIXED_VALUE
grid.status_at_node[np.isclose(elev, -9999.0)] = grid.BC_NODE_IS_CLOSED
# display the topography
grid.imshow(elev, colorbar_label="Elevation (m)", cmap="pink")
indices = np.where(elev[grid.nodes_at_right_edge] > 0.0)[0]
outlet_nodes = grid.nodes_at_right_edge[indices]
print(f"Outlet nodes: {outlet_nodes}")
print(f"Elevations of the outlet nodes: {elev[outlet_nodes]}")
links_at_outlets = grid.links_at_node[outlet_nodes]
links_to_track = links_at_outlets[
    grid.status_at_link[links_at_outlets] == grid.BC_LINK_IS_ACTIVE
].flatten()
print(f"Links at which to track discharge: {links_to_track}")
# Instantiate the component
olflow = LDOF(grid, infilt_rate=infilt_cap, roughness=n)
q = grid.at_link["water__specific_discharge"]
for i in range(num_steps):
    olflow.rain_rate = rain_rate[i]
    olflow.run_one_step(dt)
    discharge[i + 1] = np.sum(q[links_to_track]) * dx
    if i%6 == 3:
        plt.figure()
        grid.imshow(
            grid.at_node["surface_water__depth"], cmap="Blues", colorbar_label="Water depth (m)"
        )
plt.plot(time_since_storm_start / 60.0, discharge)
plt.xlabel("Time (min)")
plt.ylabel("Discharge (cms)")
plt.grid(True)
grid.imshow(
    grid.at_node["surface_water__depth"], cmap="Blues", colorbar_label="Water depth (m)"
)
olflow.uh.plot(level=1,cmap="Blues",gridLines="gray")
max(discharge)