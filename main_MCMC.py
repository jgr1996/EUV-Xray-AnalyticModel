import os
import sys
import datetime
import time
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from evolve_population import *



if __name__ == '__main__':

    # get number of cores
    number_of_cores = int(multiprocessing.cpu_count())

    # use time as label for output directory
    current_time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y_%H.%M.%S')

    observations_per_core = int(number_of_observations / number_of_cores)
    job_list = np.arange(number_of_cores)

    distribution_parameters = draw_random_distribution_parameters()

    results = Parallel(n_jobs=-1, verbose=10)(delayed(run_population)(distribution_parameters,
                                                                      observations_per_core,
                                                                      current_time_string,
                                                                      period_bias=True,
                                                                      pipeline_recovery=True,
                                                                      job_number=i) for i in job_list)

    R = []
    P = []
    for i in job_list:
        R_i = np.loadtxt("./RESULTS/{0}/R_array_{1}.csv".format(current_time_string, i), delimiter=',')
        P_i = np.loadtxt("./RESULTS/{0}/P_array_{1}.csv".format(current_time_string, i), delimiter=',')

        R = np.append(R, R_i)
        P = np.append(P, P_i)

        os.remove("./RESULTS/{0}/R_array_{1}.csv".format(current_time_string, i))
        os.remove("./RESULTS/{0}/P_array_{1}.csv".format(current_time_string, i))

    np.savetxt('./RESULTS/{0}/R_array.csv'.format(current_time_string), R, delimiter=',')
    np.savetxt('./RESULTS/{0}/P_array.csv'.format(current_time_string), P, delimiter=',')
