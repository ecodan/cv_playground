How to set up Mac OSX ARM compatible environment.

- Using up-to-date installation of Anaconda, create environment as follows (use whichever Python version is appropriate):
> CONDA_SUBDIR=osx-arm64 conda create -n [ENV_NAME] numpy python=3.9 -c conda-forge

