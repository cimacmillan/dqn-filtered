This archive contains my project code including:

  - My DQN Training code adapted from: https://github.com/dxyang/DQN_pytorch
  - My Dynamic Filter implementation (delta.py, filter_functions.py)
  - My Error-Prioritised Sampling implementation (action_buffer_non_uniform.py, sampling_functions.py)
  - A model simulation for collecting the action histograms (sim.py)
  - Jupyter Notebook plotting in "jupyter" folder for reading and interpreting training data (BCPPlotting.ipynb)
  - BCP4 scheduling and installation scripts in "jobs" folder (bcp4.job, install.job)

------------------------

Experiments can defined in main.py within the experiment_sets object.


EG. 

~~~~

experiment_sets = (
  (
    Experiment(replay_memory_size=REPLAY_BUFFER_SIZE, replay_initialisation_size=LEARNING_STARTS, repeats=5, random_seed=False, action_buffer=ActionBufferNonUniform(100000, 4, ReplayFilter(DIMENSION, 0.1, 0.1, 1, 0.1, keep), lambda x: diagonal(x), "Diag(0, 1)", True)),
  ),
  (
    Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5, random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN, ReplayFilter(DIMENSION, 0.02, 0.02, 1, 2, smallest_remove))),
  ),
)

~~~~

DQNs with Dynamic Filtering and/or Error Prioritised Sampling can be defined as experiments.
The experiment to be ran is provided with the --exp parameter when executing the main.py script (task-id corresponding to the game). EG "python main.py train --task-id 3 --exp 0"
This is built to train in parallel, so the BCP4 job script contains the number of nodes to request, with the ID being passed as the --exp parameter.
The results are saved to logs within the "figures" folder (not included).
Various plots from the data are made in the Jupyter notebook, which collects all the results from all experiments.
