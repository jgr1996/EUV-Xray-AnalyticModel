import os
import sys
import datetime
import time
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from evolve_population import *



if __name__ == '__main__':

    # get required number of observations
    number_of_observations = int(input("Please type the number of observations wanted: "))
    while type(number_of_observations) != int:
        print 'Number of observations must be an integer, please try again...'
        number_of_observations = int(input("Please type the number of observations wanted: "))

    # get number of cores
    number_of_cores = int(multiprocessing.cpu_count())

    # does user want to include transit probability
    period_bias_on = raw_input("Would you like to include transit probability? [y/n]")
    while period_bias_on not in ["y","n"]:
        print 'Please enter "y" or "n"...'
        period_bias_on = raw_input("Would you like to include transit probability? [y/n]")
    if period_bias_on == "y":
        period_bias_on = True
    elif period_bias_on == "n":
        period_bias_on = False

    # does user want to include pipeline efficiency
    pipeline_efficiency_on = raw_input("Would you like to include pipeline efficiency? [y/n]")
    while pipeline_efficiency_on not in ["y","n"]:
        print 'Please enter "y" or "n"...'
        pipeline_efficiency_on = raw_input("Would you like to include pipeline efficiency? [y/n]")
    if pipeline_efficiency_on == "y":
        pipeline_efficiency_on = True
    elif pipeline_efficiency_on == "n":
        pipeline_efficiency_on = False

    # use time as label for output directory
    current_time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y_%H.%M.%S')

    observations_per_core = int(number_of_observations / number_of_cores)
    job_list = np.arange(number_of_cores)


    Go_NoGo = raw_input("Would you like to begin? [y/n]")
    while Go_NoGo not in ["y","n"]:
        print 'Please enter "y" or "n"...'
    if Go_NoGo == "y":
        print "Let's go!"
        results = Parallel(n_jobs=-1, verbose=10)(delayed(run_population)(observations_per_core,
                                                                          current_time_string,
                                                                          period_bias=period_bias_on,
                                                                          pipeline_recovery=pipeline_efficiency_on,
                                                                          job_number=i) for i in job_list)
    elif Go_NoGo == "n":
        print 'exiting program...'
        sys.exit()

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
