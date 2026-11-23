import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import csv
import numpy as np
import sys
import statsmodels.api as sm
from scipy.optimize import least_squares


####################################################
#regression wage fit BN Kausik jan 20, 2025
#revision of Nov 2026
#convert all currency into euros;
#Estimate the numeraire for a give sigma using NLS on CES
#treats the US separately, and all others as a pool
####################################################


np.set_printoptions(precision=4, linewidth=200)
plot_flag=1
error_flag=1
xdata = np.zeros((1000,6))
c_start=np.zeros(15,dtype=int)
c_end=np.zeros(15,dtype=int)

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
for i in range (0,len(opts)):
    if opts[i]=="-p": plot_flag=int(args[i]) # toggle plot
    if opts[i]=="-e": error_flag=int(args[i]) # toggle error bar



print("plot_flag", plot_flag)
print("error_flag", error_flag)

k=0
n_country=0

#last country on the list is handled as singleton, rest are pooled
country_list=["NL","US"]
country_list=["AT","BE","DE","DK","ES","FR","IT","JP","NL","SE","UK","US"]

for c_flag in country_list:
    c_start[n_country]=k
    fname= "data/data " + c_flag + " - Sheet1.csv"
    with open(fname, 'r') as file:
        csv_reader = csv.reader(file)
        new_file=1
        for row in csv_reader:
            if(new_file):
                header=row
                new_file=0
            else:
                xdata[k]=row
                k+=1
    c_end[n_country]=k
    # adjust scale for US data
    if not ("US" in fname or "us" in fname):# scale up K if not US
        xdata[c_start[n_country]:c_end[n_country],4] *=1e3
    xdata[c_start[n_country]:c_end[n_country],1] /=xdata[c_start[n_country],1] # normalize TFP
    n_country +=1

N=c_end[n_country-1]
print("countries:",country_list)



#print(header)
xdata = xdata[0:N]

#['year', 'TFP', 'L', 'GDP', 'K', 'labor share']

#scale up GDP by 1000 for all countries
xdata[:,3]*=1000

# convert output and capital to per-labor-hour
xdata[:,3] /=xdata[:,2]
xdata[:,4] /=xdata[:,2]


#exchange rates are average across respective spans as pulled from Google Finance
# JPY jan 1995 to Dec 2020
#Others Jan 1995 to Dec 2021
for c in range(0,n_country-1):
    c_name = country_list[c]
    if c_name == 'DK': fact = 0.13404
    elif c_name == 'JP':fact = 0.00744
    elif c_name == 'SE': fact = 0.1067
    elif c_name == 'UK': fact = 1.3761
    else: continue
    c_s=c_start[c];c_e =c_end[c]
    xdata[c_s:c_e,3] *= fact
    xdata[c_s:c_e,4] *= fact


#handy arrays
ls = xdata[:,5] # labor share
k = xdata[:,4]   #capital to labor ratio
y_adj = (xdata[:,3]/xdata[:,1]) #output-per-labor-hour over TFP


def ces_rhs(params, lam, k, sigma):
    """
    Evaluates: b0 + (1/rho) * ln(lam + (1 - lam) * (k/k0)^rho) where rho = (1 - sigma) / sigma.
    """
    b0, k0 = params
    rho = (1.0 - sigma) / sigma

    if abs(rho) < 1e-6:
        print("small rho", rho)
        rho = 1.e-6

    return b0 + (1/rho)*np.log(lam + (1 - lam)*np.power(k/k0, rho))

def cobb_douglas_rhs(params, lam, k):
    """
    Evaluates: b0 + 1/(1-lam)*ln(k/k0)
    """
    b0, k0 = params

    return b0 + (1/1-lam)*np.log(k/k0)


def fit_nls(y_adj, lam, k, sigma, clusters=None):
    """
    Fits CES/cobb-douglas via Non-Linear Least Squares using pure NumPy/SciPy.

    Parameters:
      - y_adj:    1D array of (Y / (L * A)
      - lam:      1D array of labor share (wL / Y)
      - k:        1D array of (K/ L)
    """

    def residuals(params):
        if sigma == 1.0:
            return np.log(y_adj) - cobb_douglas_rhs(params, lam, k) # for cobb-douglas
        else:
            return np.log(y_adj) - ces_rhs(params, lam, k, sigma) #for CES

    # Bound k0 > 0.01 to avoid division by zero
    res = least_squares(residuals, [1.0, 10.0], bounds=([-np.inf, 0.01], [np.inf, np.inf]))

    b0_hat, k0_hat = res.x


    # Extract Jacobian and degrees of freedom
    J = res.jac
    dof = res.fun.size - res.x.size  # number of data points - number of parameters

    # Compute residual variance (reduced chi-squared)
    residual_variance = np.sum(res.fun**2) / dof

    # Compute covariance matrix and standard errors (1-sigma)
    cov = np.linalg.inv(J.T @ J) * residual_variance
    standard_error = np.sqrt(np.diag(cov))

    return {
        'sigma': sigma,
        'b0': b0_hat,
        'b0_SE':standard_error[0],
        'k0': k0_hat,
        'k0_SE':standard_error[1],
        'n': J.shape[0],
        'converged': res.success,
        'iters': res.nfev
    }



#for pool countries
print("pool ######################")
sigma=0.99
c_s, c_e =0, c_end[n_country-2]
results = fit_nls(y_adj[c_s:c_e], ls[c_s:c_e], k[c_s:c_e], sigma)
for key, v in results.items():
    print(f"{key:14s}: {v:.4f}" if isinstance(v, float) else f"{key:14s}: {v}")


#for US
c_s, c_e = c_start[n_country-1], c_end[n_country-1]
print("US ######################")
for sigma in [0.92, 1.0]:
    results = fit_nls(y_adj[c_s:c_e], ls[c_s:c_e], k[c_s:c_e], sigma)
    for key, v in results.items():
        print(f"{key:14s}: {v:.4f}" if isinstance(v, float) else f"{key:14s}: {v}")


