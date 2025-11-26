include(DuneSphinxDoc)
add_sphinx_targets(sphinx)
dune_symlink_to_source_files(FILES
additional.rst index.rst contributions.rst
landlab_descr.rst twophaseflow_descr.rst ddfem_descr.rst
mmesh_descr.rst todo.rst citing.rst
corepy.rst installation.rst overview.rst
stokes.rst furtherexamples.rst nextsteps.rst furthertopics.rst
inforesources.rst userprojects.rst
gridviews_and_adaptivity.rst cpp.rst developers.rst slurmbatchscript.rst
changelog.rst changelog290.rst changelog210.rst
scheme_api.rst
3dexample.py          spiral.py
crystal.py            lineplot.py           svg2pdf.py            cppfunctions.py
dune-corepy.py        mcf-algorithm.py      twophaseflow.py       chemical.py
concepts.py           dune-fempy.py         mcf_cmp_plot.py       euler.py
elasticity.py         mcf.py
uzawa-scipy.py        monolithicStokes.py   fieldsplitStokes.py  cylinder.py
othergrids.py         gitlab-formatting.py  gmsh2dgf.py          pandoc-formatting.py
vemdemo.py            chimpl.py             discontinuousgalerkin.py
interpolation.py      wave.py               laplace-adaptive.py   laplace-dwr.py laplace-dwr-algorithm.py
backuprestore.py      parallelization.py    boundary.py           filteredgridview.py
solversInternal.py    solversExternal.py    evalues_laplace.py    biharmonic_IPC0.py    mixed_poisson.py    mmesh.py
overview_of_advection_solver.py linear_diffusion_overland_flow_dune.py
crystal_nb.ipynb           mcf_nb.ipynb               chemical_nb.ipynb
dune-corepy_nb.ipynb       spiral_nb.ipynb            euler_nb.ipynb
concepts_nb.ipynb          dune-fempy_nb.ipynb        twophaseflow_nb.ipynb
elasticity_nb.ipynb        uzawa-scipy_nb.ipynb       othergrids_nb.ipynb
laplace-adaptive_nb.ipynb  vemdemo_nb.ipynb           chimpl_nb.ipynb
discontinuousgalerkin_nb.ipynb gmsh2dgf_nb.ipynb
monolithicStokes_nb.ipynb  fieldsplitStokes_nb.ipynb  cylinder_nb.ipynb
laplace-dwr_nb.ipynb       laplace-dwr-algorithm_nb.ipynb
lineplot_nb.ipynb          wave_nb.ipynb
mcf-algorithm_nb.ipynb     parallelization_nb.ipynb   backuprestore_nb.ipynb
boundary_nb.ipynb          mmesh_nb.ipynb             mixed_poisson_nb.ipynb
solversInternal_nb.ipynb   solversExternal_nb.ipynb
cppfunctions_nb.ipynb      laplace-dwr.hh             utility.hh
uzawa.hh                   evalues_laplace_nb.ipynb   biharmonic_IPC0_nb.ipynb
overview_of_advection_solver_nb.ipynb linear_diffusion_overland_flow_dune_nb.ipynb
levelgridview_nb.rst geoview_nb.rst geoview_nb_files
filteredgridview_nb.ipynb
dune-python-logo.png
figures _static dune-fempy.bib .special.rst
horizontal.msh 3dmesh.dgf soap.dgf sphere.dgf triangle.dgf unitcube-2d.dgf wave_tank.msh quads.msh hugo_site.asc
config.opts
mcf.gif
favicon.ico
)

add_subdirectory("_templates")

#add_custom_target(compilegrids2 ${PYTHON_EXECUTABLE} compilegrids.py 2)
#add_custom_target(compilegrids3 ${PYTHON_EXECUTABLE} compilegrids.py 3)
