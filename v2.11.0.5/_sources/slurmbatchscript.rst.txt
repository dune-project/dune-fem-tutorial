##################
Slurm batch script 
##################

.. _slurmbatchscript:

On a cluster where multiple parallel jobs are run simultaneously, it's
advisable to use one separate cache per job. This can be easily
done by copying an existing cache with pre-compiled modules as demonstrated
below.

.. code-block:: bash

  #!/bin/bash
  #
  # Replace the fields denoted by <SOME-VALUE-YOU-SHOULD-REPLACE>
  #
  #SBATCH -A <PROJECT-ID>
  #SBATCH -t 00:10:00
  #SBATCH -J <JOB-NAME>
  #SBATCH -o logs/%j.out
  #SBATCH -e logs/%j.err
  #
  # Number of nodes and threads the job should use
  #SBATCH -N 1
  #SBATCH --tasks-per-node=4
  #SBATCH --cpus-per-task=2

  export DUNE_NUM_THREADS=2

  ###
  # Warning! This template script assumes:
  # * your Dune installation is in `$DUNE_CONTROL_PATH`
  # * you installed Dune using the `dune-fem-dg/scripts/build-dune-fem-dg.sh` script
  # * you want to create the `$RUNDIR` directory to store job data
  # * This should also work fine with other installations, i.e. pip install but
  # * might require some adjustments.
  ###
  RUNDIR="`pwd`/jobs/$SLURM_JOB_ID"

  # Create `rundir` directory where the job will run.
  # The rundir stores the runscript (this script) + any files written by the job.
  mkdir -p "$RUNDIR"
  cat "$0" > "$RUNDIR/run.sh"
  cd "$RUNDIR"

  : '
  Optional comment describing the purpose of the run.
  '

  # activate a previously created virtual environment
  source "$DUNE_CONTROL_PATH/dune-env/bin/activate"

  # Use local compilation cache.
  # To use the global cache - comment the following lines.

  # extract last folder name from DUNE_PY_DIR
  CACHEFOLDER=${DUNE_PY_DIR##*/}
  # Use local compilation cache to avoid clash with other jobs.
  # To use the global cache - comment the following three lines.
  cp -r $DUNE_PY_DIR $RUNDIR
  # Reset the DUNE_PY_DIR to the new directory
  export DUNE_PY_DIR="$RUNDIR/$CACHEFOLDER"
  # trigger a re-configuration to adjust paths within DUNE_PY_DIR
  rm "$DUNE_PY_DIR/dune-py/.noconfigure"

  echo "JOB STARTED"

  # Example, replace this part with the test case you intended to run.
  mpirun \
  python $DUNE_CONTROL_PATH/testcase.py

  echo "JOB FINISHED"
