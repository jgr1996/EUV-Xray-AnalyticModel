from __future__ import division
import numpy as np
from scipy import special
from numpy import random as rand
import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
from sklearn.neighbors import KernelDensity



def B(x, order, coefficients):

    coefficients = np.array(coefficients)
    poly_array = np.array([special.binom(order, i)*(x**i)*((1-x)**(order-i)) for i in range(order+1)])
    B = np.dot(coefficients, poly_array)

    return B

def priors(order):
    N = 0
    x_range = np.arange(0.0, 1.0001, 0.0001)
    a_range = np.arange(0.0, 1.0001, 0.0001)

    Xrange = np.logspace(-4,0,200)
    Mrange = np.logspace(np.log10(0.5),np.log10(15),200)

    while N < 10000:

        coeffs = rand.uniform(low=0.0,high=50.0,size=(2*order)+2)

        M_coeffs = coeffs[order+1:2*(order+1)]
        M_min, M_max = np.log10(0.5), np.log10(15.0)
        M_poly_min, M_poly_max = B(0, order, M_coeffs), B(1, order, M_coeffs)
        M_poly_norm = M_poly_max - M_poly_min
        M_norm = M_max - M_min
        CDF_M = [(((M_norm/M_poly_norm) * ((B(i, order, M_coeffs)) - M_poly_min)) + M_min) for i in a_range]
        for i in range(1,len(CDF_M)):
            if CDF_M[i-1] > CDF_M[i]:
                continue

        X_coeffs = coeffs[0:order+1]
        X_min, X_max = -4.0, 0.0
        X_poly_min, X_poly_max = B(0, order, X_coeffs), B(1, order, X_coeffs)
        X_poly_norm = X_poly_max - X_poly_min
        X_norm = X_max - X_min
        CDF_X = [(((X_norm/X_poly_norm) * ((B(i, order, X_coeffs)) - X_poly_min)) + X_min) for i in a_range]
        for i in range(1,len(CDF_X)):
            if CDF_X[i-1] > CDF_X[i]:
                continue


        measurements = []
        for j in range(1000):
            U = rand.uniform()
            measurements.append(((M_norm/M_poly_norm) * ((B(U, order, M_coeffs)) - M_poly_min)) + M_min)
        measurements = np.array(measurements)
        kde = KernelDensity(kernel='gaussian',bandwidth=0.1).fit(measurements[:,None])
        logprob = kde.score_samples(np.log10(Mrange[:,None]))
        plt.figure(1)
        plt.plot(Mrange,np.exp(logprob), alpha=0.01, color='black')
        plt.xscale('log')
        plt.xlabel(r'M$_{core}$ [M$_\oplus$]', fontsize=14)




        measurements = []
        for j in range(1000):
            U = rand.uniform()
            measurements.append((((X_norm/X_poly_norm) * ((B(U, order, X_coeffs)) - X_poly_min)) + X_min))
        measurements = np.array(measurements)
        kde = KernelDensity(kernel='gaussian',bandwidth=0.23).fit(measurements[:,None])
        logprob = kde.score_samples(np.log10(Xrange[:,None]))
        plt.figure(2)
        plt.plot(Xrange,np.exp(logprob), alpha=0.01, color='black')
        plt.xscale('log')
        plt.xlabel(r'Initial Envelope Mass Fraction X', fontsize=14)


        N = N + 1
        print N

    plt.show()



def plot_pdfs(order, coeffs, plot_cdf=False):

    x_range = np.arange(0.0, 1.0001, 0.0001)


    M_coeffs = coeffs[order+1:2*(order+1)]
    M_min, M_max = np.log10(0.5), np.log10(15)
    M_poly_min, M_poly_max = B(0, order, M_coeffs), B(1, order, M_coeffs)
    M_poly_norm = M_poly_max - M_poly_min
    M_norm = M_max - M_min

    X_coeffs = coeffs[0:order+1]
    X_min, X_max = -4.0, 0.0
    X_poly_min, X_poly_max = B(0, order, X_coeffs), B(1, order, X_coeffs)
    X_poly_norm = X_poly_max - X_poly_min
    X_norm = X_max - X_min

    print "Coeffs for M CDF:", M_coeffs
    print "Coeffs for X CDF:", X_coeffs


    plt.figure(1)
    Mrange = np.logspace(np.log10(0.5),np.log10(15),200)
    # np.savetxt("../../../Computing/IDL plots/bernstein/distributions/M.csv", Mrange, delimiter=',')
    for i in range(100):
        N_range = np.arange(1000)
        measurements = []
        for j in range(len(N_range)):
            U = rand.uniform()
            measurements.append(((M_norm/M_poly_norm) * ((B(U, order, M_coeffs)) - M_poly_min)) + M_min)
        measurements = np.array(measurements)
        kde = KernelDensity(kernel='gaussian',bandwidth=0.1).fit(measurements[:,None])
        logprob = kde.score_samples(np.log10(Mrange[:,None]))
        plt.plot(Mrange,np.exp(logprob), alpha=0.02, color='black')
    plt.xlabel(r'M$_{core}$ [M$_\oplus$]', fontsize=14)
    plt.xscale('log')
    # np.savetxt("../../../Computing/IDL plots/bernstein/distributions/pdf_{0}.csv".format(i), np.exp(logprob), delimiter=',')


    plt.figure(2)
    Xrange = np.logspace(-4,0,200)
    # np.savetxt("../../../Computing/IDL plots/bernstein/distributions/X.csv", Xrange, delimiter=',')
    for i in range(100):
        N_range = np.arange(1000)
        measurements = []
        for j in range(len(N_range)):
            U = rand.uniform()
            measurements.append((((X_norm/X_poly_norm) * ((B(U, order, X_coeffs)) - X_poly_min)) + X_min))
        measurements = np.array(measurements)
        kde = KernelDensity(kernel='gaussian',bandwidth=0.23).fit(measurements[:,None])
        logprob = kde.score_samples(np.log10(Xrange[:,None]))
        # np.savetxt("../../../Computing/IDL plots/bernstein/distributions/pdf_{0}.csv".format(i), np.exp(logprob), delimiter=',')
        plt.plot(Xrange, np.exp(logprob), alpha=0.02, color='black')
    plt.xlabel(r'Initial Envelope Mass Fraction X', fontsize=14)
    plt.xscale('log')

    if plot_cdf:
        x_range = np.arange(0.0, 1.0001, 0.0001)

        plt.figure(3)
        plt.plot(x_range, [(((M_norm/M_poly_norm) * ((B(i, order, M_coeffs)) - M_poly_min)) + M_min) for i in x_range], linewidth=3)
        plt.ylabel(r"CDF$_M$", fontsize=18)

        plt.figure(4)
        plt.plot(x_range, [(((X_norm/X_poly_norm) * ((B(i, order, X_coeffs)) - X_poly_min)) + X_min) for i in x_range], linewidth=3)
        plt.ylabel(r"log( CDF$_X$ )", fontsize=18)


    plt.show()





# priors(5)
plot_pdfs(5, [0.0,13.0,0.0,15.79,14.5,20.63,35.0,0.0,0.0,2.37,43.0,32.15,50.0], plot_cdf=True)
