import sys
sys.path.append("./OpenOA/examples")
import operational_analysis
print("OpenOA Version: ", operational_analysis.__file__, operational_analysis.__version__)
from operational_analysis.methods import plant_analysis
from project_ENGIE import Project_Engie
from tqdm import tqdm
import pandas as pd
import itertools
import time
import multiprocessing
import tqdm
import logging
import random
import numpy as np

def experiment(options):

    num_sim,regression_frac,qmc,repetition,seed = options
    start_time = time.perf_counter()

    pa = plant_analysis.MonteCarloAEP(project,
                                    reanal_products = ['era5', 'merra2'],
                                    regression_frac=regression_frac,
                                    qmc=qmc,
                                    seed=seed)
    pa.run(num_sim=num_sim, reanal_subset=['era5', 'merra2'])

    total_time = time.perf_counter() - start_time

    mean,std = pa.results["aep_GWh"].mean(), pa.results["aep_GWh"].std()

    res = {
        "num_sim":num_sim,
        "regression_frac":regression_frac,
        "qmc":qmc,
        "repetition": repetition,
        "time": total_time,
        "AEP mean": mean,
        "AEP stdev": std,
    }

    return res

def redirect_output():
    name = multiprocessing.current_process().name
    sys.stdout = open("log/"+name+".stdout.log", "w")
    sys.stderr = open("log/"+name+".stderr.log", "w")
    log = logging.getLogger("operational_analysis.methods.plant_analysis")
    log.setLevel(logging.ERROR)

project = Project_Engie('./OpenOA/examples/data/la_haute_borne')
project.prepare()

num_sim = [128,256,512]
regression_frac = [0.5, 0.75, 1.0] #[0.1, 1.0]
qmc = [False, True]
repetitions = range(30)

experiment_configs = list(itertools.product(num_sim,regression_frac,qmc,repetitions))
experiment_configs = [x + (random.randint(0,999999),) for x in experiment_configs]

with multiprocessing.Pool(6, initializer=redirect_output) as p:
    results = list(tqdm.tqdm(p.imap_unordered(experiment, experiment_configs), total=len(experiment_configs)))

results = list(results)
df = pd.DataFrame(results)
df.to_csv("experiment1.csv", index=False)
