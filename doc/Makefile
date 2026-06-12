SHELL := /bin/bash
PATH := bin:$(PATH)

FIGURES = figures/3dexample.png figures/interpolation_discrete.png figures/interpolation_exact.png figures/interpolation_error.png
NBS = spiral_nb.ipynb elasticity_nb.ipynb wave_nb.ipynb\
      uzawa-scipy_nb.ipynb monolithicStokes_nb.ipynb fieldsplitStokes_nb.ipynb\
      cylinder_nb.ipynb\
      laplace-adaptive_nb.ipynb crystal_nb.ipynb mmesh_nb.ipynb\
      cppfunctions_nb.ipynb evalues_laplace_nb.ipynb\
      biharmonic_IPC0_nb.ipynb mixed_poisson_nb.ipynb\
			mcf_nb.ipynb boundary_nb.ipynb \
      solversInternal_nb.ipynb solversExternal_nb.ipynb \
			mcf-algorithm_nb.ipynb lineplot_nb.ipynb laplace-dwr_nb.ipynb \
      laplace-dwr-algorithm_nb.ipynb othergrids_nb.ipynb \
			concepts_nb.ipynb dune-fempy_nb.ipynb backuprestore_nb.ipynb parallelization_nb.ipynb \
      dune-corepy_nb.ipynb discontinuousgalerkin_nb.ipynb \
      twophaseflow_nb.ipynb euler_nb.ipynb chemical_nb.ipynb \
			geoview_nb.ipynb geoview_nb.rst \
			levelgridview_nb.ipynb levelgridview_nb.rst \
      filteredgridview_nb.ipynb \
      vemdemo_nb.ipynb chimpl_nb.ipynb \
      overview_of_advection_solver_nb.ipynb \
      linear_diffusion_overland_flow_dune_nb.ipynb
			# spiral_adapt_nb.ipynb 


MODULES = spiral.py.jitmodules elasticity.py.jitmodules wave.py.jitmodules\
      uzawa-scipy.py.jitmodules monolithicStokes.py.jitmodules fieldsplitStokes.py.jitmodules\
      cylinder.py.jitmodules\
      laplace-adaptive.py.jitmodules crystal.py.jitmodules mmesh.py.jitmodules\
      cppfunctions.py.jitmodules evalues_laplace.py.jitmodules\
      biharmonic_IPC0.py.jitmodules mixed_poisson.py.jitmodules\
			mcf.py.jitmodules boundary.py.jitmodules \
      solversInternal.py.jitmodules solversExternal.py.jitmodules \
			mcf-algorithm.py.jitmodules lineplot.py.jitmodules laplace-dwr.py.jitmodules \
      laplace-dwr-algorithm.py.jitmodules othergrids.py.jitmodules \
			concepts.py.jitmodules dune-fempy.py.jitmodules backuprestore.py.jitmodules parallelization.py.jitmodules \
      dune-corepy.py.jitmodules discontinuousgalerkin.py.jitmodules \
      twophaseflow.py.jitmodules euler.py.jitmodules chemical.py.jitmodules \
			geoview.py.jitmodules \
			levelgridview.py.jitmodules \
      filteredgridview.py.jitmodules \
      vemdemo.py.jitmodules chimpl.py.jitmodules

# .EXPORT_ALL_VARIABLES:

export DUNE_LOGMODULES=8
export DUNE_FORCE_BUILD=1
export DUNE_LOG_LEVEL=ERROR
export ALUGRID_VERBOSITY_LEVEL=0
export IPYTHONDIR=.
# ignore SAVE_BUILD since this will appear in notebooks
export DUNE_SAVE_BUILD=none
# ignore Python warnings since these will appear in notebooks
export PYTHONWARNINGS=ignore

all: $(NBS) $(FIGURES)
MODULES : $(MODULES)

.PHONY: clean distclean linkcheck

linkcheck:
	@sphinx-build -b linkcheck . html

clean:
	@rm -f *.vtu *.pvtu *.p $(FIGURES) $(NBS) $(MODULES)
	@rm -f geoview_nb_files/geoview_nb_1_0.png
distclean: clean
	@rm -f *.bbl
	@rm -rf html

.SECONDARY:

%.py.jitmodules: %.py
	DUNEPY_DISABLE_PLOTTING=1 DUNE_LOGMODULES=4 python $<

%_nb.ipynb: %.py
	@jupytext $< --execute --set-kernel python3 --output $*_nb.ipynb

%_nb.ipynb: %.ipynb
	@jupyter nbconvert --execute --to notebook --output $*_nb.ipynb $<

%_nb.rst: %_nb.ipynb
	@jupyter-nbconvert $< --to rst

figures/3dexample.png: 3dexample.py dune-fempy_nb.ipynb
	@pvpython 3dexample.py

figures/interpolation_discrete.png figures/interpolation_exact.png figures/interpolation_error.png &: interpolation.py dune-corepy_nb.ipynb
	@pvpython interpolation.py
