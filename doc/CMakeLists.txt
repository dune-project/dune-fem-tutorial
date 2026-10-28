include(DuneSphinxDoc)
add_sphinx_targets(sphinx)

set(bases
crystal            lineplot           cppfunctions       biharmonic_IPC0
dune-corepy        mcf-algorithm      twophaseflow       chemical
concepts           dune-fempy         euler
elasticity         mcf                othergrids         mmesh
uzawa-scipy        monolithicStokes   fieldsplitStokes   cylinder
vemdemo            chimpl             discontinuousgalerkin
spiral             wave               laplace-adaptive   laplace-dwr laplace-dwr-algorithm
backuprestore      parallelization    boundary           filteredgridview
solversInternal    solversExternal    evalues_laplace    mixed_poisson
parallelrun        overview_of_advection_solver linear_diffusion_overland_flow_dune
)

foreach(base IN LISTS bases)
  list(APPEND files
    "${base}.py"
    "${base}_nb.ipynb"
    "${base}_script.py"
  )
endforeach()
list(APPEND files
3dexample interpolation
gitlab-formatting.py pandoc-formatting.py svg2pdf.py
landlab_descr.rst twophaseflow_descr.rst ddfem_descr.rst
mmesh_descr.rst todo.rst citing.rst index.rst contributions.rst
corepy.rst installation.rst overview.rst
stokes.rst furtherexamples.rst nextsteps.rst furthertopics.rst
inforesources.rst userprojects.rst versions.rst
gridviews_and_adaptivity.rst cpp.rst developers.rst slurmbatchscript.rst
changelog.rst changelog290.rst changelog210.rst changelog212.rst
levelgridview_nb.rst geoview_nb.rst geoview_nb_files scheme_api.rst
utility.hh uzawa.hh laplace-dwr.hh
dune-python-logo.png figures _static dune-fempy.bib .special.rst
horizontal.msh 3dmesh.dgf soap.dgf sphere.dgf triangle.dgf unitcube-2d.dgf
wave_tank.msh quads.msh hugo_site.asc
config.opts mcf.gif favicon.ico
crystal03_0.png  crystal13_0.png  crystal23_0.png  crystal3_0.png
crystal03_1.png  crystal13_1.png  crystal23_1.png  crystal3_1.png
crystal03_2.png  crystal13_2.png  crystal23_2.png  crystal3_2.png
crystal03_3.png  crystal13_3.png  crystal23_3.png  crystal3_3.png
crystal03_4.png  crystal13_4.png  crystal23_4.png  crystal3_4.png
crystal03_5.png  crystal13_5.png  crystal23_5.png  crystal3_5.png
wave_mpi0.png    wave_mpi1.png
)

dune_symlink_to_source_files(FILES ${files})

add_subdirectory("_templates")
