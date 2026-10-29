This repository contains computations for LED-KDE. 

Run the script using run.py. Important parameters (extract from help in run.py):

```shell
--method {led,kde,led_retain}
                        method of sampling. Choose led for LED-KDE, kde for KDE and led_retain with LED-KDE in a data retaining scheme
--n N                 number of iterations
--centers CENTERS [CENTERS ...]
                        centers of the initial mixture distribution
--scales SCALES [SCALES ...]
                        if dist_type is gaussian, then scales (sddevs) of gaussians. Irrelevant if dist_type is uniform. Can be a float or a list of
                        floats the same legth as centers
--widths WIDTHS [WIDTHS ...]
                        if dist_type is uniform, then widths of uniforms. Irrelevant if dist_type is gaussian. Can be a float or a list of floats the
                        same legth as centers.
--t_scale T_SCALE
--k_scale K_SCALE
--n_samples N_SAMPLES
                        Number of samples chosen each round. Choose a small value (~50, 100) if method = led_retains
--n_init_samples N_INIT_SAMPLES
                        number of initial samples, only relevant for data retaining scheme
--verbose if True will print statistics along the way, if False will just show a progress bar
```


For explanations of all parameters run
```python
python3 run.py --help
```

Structure of a repository:
```
run.py - main file which is a front to run experiments, calling scripts from src module
tests.py - some unit tests. There is no 100% test coverage though. If run with --profil will run a simple profiling
src/
  kde.py - a file which contains computations of KDE, alpha function in LED-KDE and samplers
  pipelines.py - pipelines that prepare necessary objects to run functions from kde.py
  distributions.py - objects that prepare initial distributions. For now mixtures of gaussians and uniforms
  logops.py - helper functions that deal with logging results
  tools.py - some tools that are used in at least one place
```

Example scripts:
```shell
example_1d.sh # simple LED-retain
run_suite_led.sh # a bunch of LED-forget scripts
run_suite_retain.sh # a bunch of LED-remain scripts 
speed_tests.sh # pipelines with gaussians with some parameters to test how fast things are going to run on your machine
```
