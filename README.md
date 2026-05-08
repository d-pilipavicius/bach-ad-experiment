# Bachelor's experiment

## Fetch repository

Fetch the current repository using
```bash
git clone https://github.com/d-pilipavicius/bach-ad-experiment.git
```

And after the repository is downloaded, run
```bash
cd bach-ad-experiment
```

## Set up environment
All dependencies used in this project are available inside the [environment.yml](environment.yml) file.

In order to automatically set up the python environment, use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) or any other Anaconda like distributions.

To build an environment for execution using conda, run

```bash
conda env create -f environment.yml
```

To use the built environment, run 
```bash
conda activate anomalib_dev
```

## Running the code

TODO: How to run experiments


## Remove the environment after experimenting

When you are done with experimenting, run in order to remove the environment from your computer
```bash
conda env remove -n anomalib_dev
```