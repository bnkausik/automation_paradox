import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import csv
import numpy as np
import sys
import statsmodels.api as sm


####################################################
#regression wage fit BN Kausik jan 20, 2025
#revision of April 7 2025 via Taylor series OLS
#convert all currency into euros;
#ordinary least squares with clustering standard error
#treats the US separately, and all others as a pool
####################################################


np.set_printoptions(precision=4, linewidth=200)
plot_flag=1
error_flag=0
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
country_list=["AT","BE","DE","DK","ES","FR","IT","JP","NL","SE","UK","US"]

for c_flag in country_list:
    c_start[n_country]=k
    fname="data/data "+c_flag+" - Sheet1.csv"
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


"""
"""
#nominal conversion to euro
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
ls=xdata[:,5] # labor share
k=xdata[:,4]   #capital to labor ratio
print("Pool mink, meank, maxk", np.min(k[:c_end[n_country-2]]),np.mean(k[:c_end[n_country-2]]),np.max(k[:c_end[n_country-2]]))
c_s=c_start[n_country-1];c_e =c_end[n_country-1]
print("US mink, meank, maxk", np.min(k[c_s:c_e]),np.mean(k[c_s:c_e]),np.max(k[c_s:c_e]))

#regression LHS
log_out=np.log(xdata[:,3]/xdata[:,1]) - (1-ls)*np.log(k)

# observed and predicted wages for comparsion
log_w =np.log(xdata[:,5]*xdata[:,3])
pred_log_w =np.zeros(N)


X=np.ones((N,2))     # regression variables
groups = np.zeros(N)

#pool together non US countries
for country in range(0,n_country-1):
    c_s=c_start[country];c_e =c_end[country]
    X[c_s:c_e,0]=0.5*(1-ls[c_s:c_e])*ls[c_s:c_e]*np.power(np.log(k[c_s:c_e]),2)
    groups[c_s:c_e]=country

c_s=0;c_e=c_end[n_country-2]
if n_country-2 >0:
    pool_model = sm.OLS(log_out[c_s:c_e], X[c_s:c_e]).fit(cov_type='cluster', cov_kwds = {"groups":groups[c_s:c_e]})
else:
    pool_model = sm.OLS(log_out[c_s:c_e], X[c_s:c_e]).fit()   #default covariance
for country in range(0,n_country-1):
    c_s=c_start[country];c_e =c_end[country]
    pred_log_w[c_s:c_e] = (pool_model.predict(X[c_s:c_e]) + np.log(ls[c_s:c_e]) +
                       (1-ls[c_s:c_e])*np.log(k[c_s:c_e])+np.log(xdata[c_s:c_e,1]))
print(pool_model.summary())



#country-specific fit for the US
country=n_country-1
c_s=c_start[country];c_e =c_end[country]
X[c_s:c_e,0]=0.5*(1-ls[c_s:c_e])*ls[c_s:c_e]*np.power(np.log(k[c_s:c_e]),2)
us_model = sm.OLS(log_out[c_s:c_e], X[c_s:c_e]).fit()
pred_log_w[c_s:c_e] = (us_model.predict(X[c_s:c_e]) + np.log(ls[c_s:c_e]) +
                       (1-ls[c_s:c_e])*np.log(k[c_s:c_e])+np.log(xdata[c_s:c_e,1]))
print(us_model.summary())


print("#####################################################")

gamma_pool = np.zeros(3)
sigma_pool = np.zeros(3)
gamma_pool[1] = -pool_model.params[0]
gamma_pool[0] = gamma_pool[1] - pool_model.bse[0]
gamma_pool[2] = gamma_pool[1] + pool_model.bse[0]
for i in range(3): sigma_pool[i] = 1/(1-gamma_pool[i])
print("gamma_pool",gamma_pool, pool_model.bse[0])
print("sigma_pool, SE",sigma_pool, max(np.abs(sigma_pool-sigma_pool[1])))

print("\n\n\n")

gamma_us = np.zeros(3)
sigma_us = np.zeros(3)
gamma_us[1] = -us_model.params[0]
gamma_us[0] = gamma_us[1] - us_model.bse[0]
gamma_us[2] = gamma_us[1] + us_model.bse[0]
for i in range(3): sigma_us[i] = 1/(1-gamma_us[i])
print("gamma_us, SE",gamma_us, us_model.bse[0])
print("sigma_us, SE",sigma_us, max(np.abs(sigma_us-sigma_us[1])))


#estimate mean derivatives dln(w)/dln(lambda)
derivs=np.zeros((n_country,3,3))     # (country, low/mean/high, direct/indirect/net)
SE=np.zeros(n_country)
for c in range(0,n_country):
    c_s=c_start[c];c_e =c_end[c]
    for i in range(3):
        if c<n_country-1:
            gamma=gamma_pool[i]
        else:
            gamma=gamma_us[i]
        derivs[c,i,0] = 1      # direct effect
        derivs[c,i,1] = np.mean(  #indirect effect
            - ls[c_s:c_e]*np.log(k[c_s:c_e])*
            (1 + 0.5*gamma*(1-2*ls[c_s:c_e])*np.log(k[c_s:c_e])
            ))
        derivs[c,i,2] = derivs[c,i,0]+derivs[c,i,1]   #net effect

    SE[c] = max(np.abs(derivs[c,1,2]-derivs[c,0,2]),np.abs(derivs[c,1,2]-derivs[c,2,2]))


deriv_avg = derivs[:,1,:]



print("\n\nnegative of dln(w)/dln(lambda)\n",-deriv_avg)
print("standard error", SE)


order =np.argsort(-deriv_avg[:,2]) # plot in order of increasing net derivative
c_list=[country_list[i] for i in order]
fig, ax = plt.subplots()
ax.axhline(0.0,color='gray', linewidth=0.5)
m_size= 10 # marker size

plt.scatter(c_list, -deriv_avg[order,0], label="Direct",c='#4285f4',s=m_size)

plt.scatter(c_list, -deriv_avg[order,1] ,label="Indirect",c='#ea4335',s=m_size,marker='s')
#plt.errorbar(c_list, -deriv_avg[order,1],yerr=1.96*SE[order],fmt='none', ecolor='#ea4335')

plt.scatter(c_list, -deriv_avg[order,2], label="Net", c='#fbbc04',s=m_size,marker='D')
plt.errorbar(c_list, -deriv_avg[order,2],yerr=1.96*SE[order],fmt='none',ecolor='#fbbc04')

plt.legend(loc="upper left")
plt.ylabel("Derivative",fontsize=14)
#plt.title("wage derivate with pool))
plt.savefig("series_wage_derivative.pdf", format="pdf")

plt.figure(4)
plt.plot(np.arange(0,N),log_w)
plt.plot(np.arange(0,N),pred_log_w)
"""
"""




#This section plots wage growth contributions by source
w_g=np.zeros((n_country,5))

for c in range(0,n_country):
    c_s=c_start[c]; c_e=c_end[c]
    if c<n_country-1:
        gamma=gamma_pool[1]
    else:
        gamma=gamma_us[1]

    #TFP contribution to wage growth
    w_g[c,1] = (np.log(xdata[c_e-1,1]/xdata[c_s,1]))/(c_e-c_s)


    #k contribution as mean of forward and backward differences
    w_g[c,0] = np.mean(
                0.5*(np.log(k[c_s+1:c_e]/k[c_s:c_e-1]))*
                (1-ls[c_s:c_e-1])*(1- gamma*ls[c_s:c_e-1]*np.log(k[c_s:c_e-1])) +
                0.5*(np.log(k[c_s+1:c_e]/k[c_s:c_e-1]))*
                (1-ls[c_s+1:c_e])*(1- gamma*ls[c_s+1:c_e]*np.log(k[c_s+1:c_e]))
                        )


    #labor share contribution as mean of forward and backward difference
    w_g[c,2]= np.mean(
            0.5*(np.log(ls[c_s+1:c_e]/ls[c_s:c_e-1]))*(1 -
                ls[c_s:c_e-1]*np.log(k[c_s:c_e-1])* (1+ 0.5*gamma*(1 - 2*ls[c_s:c_e-1])*np.log(k[c_s:c_e-1]))) +
            0.5*(np.log(ls[c_s+1:c_e]/ls[c_s:c_e-1]))*(1 -
            ls[c_s+1:c_e]*np.log(k[c_s+1:c_e])* (1+ 0.5*gamma*(1 - 2*ls[c_s+1:c_e])*np.log(k[c_s+1:c_e]))))

    # calculate from wage deriv to verify
    #w_g[c,2] = deriv_avg[c,2]*(np.log(ls[c_e-1])-np.log(ls[c_s]))/(c_e-c_s)

    #observed total wage growth
    w_g[c,3] = (log_w[c_e-1] - log_w[c_s])/(c_e-c_s)

    # error between observed total and estimated wage growth
    w_g[c,4] = np.sum(w_g[c,0:3]) - w_g[c,3]

w_g *=100 #convert to percentages
if error_flag: w = 0.15 # bar width and spacing
else: w=0.2

"""
#plot all countries separately
order=np.argsort(w_g[:,3])
x=np.arange(n_country)
plt.figure(2)
plt.bar(x-1.5*w, w_g[order,2],w, color='#4285f4')
plt.bar(x -0.5*w, w_g[order,1],w,color='#ea4335')
plt.bar(x+0.5*w, w_g[order,0],w, color='#fbbc04')
plt.bar(x+1.5*w, w_g[order,3],w, color="green")
if error_flag: plt.bar(x+2.5*w, w_g[order,4],w, color="purple")
plt.xticks(x, [country_list[i] for i in order])
plt.legend(["Labor share", "TFP","K/L","Observed Total","Error"], loc="upper left")
plt.ylabel("mean Annual Real Wage Growth (%)")
plt.savefig("wage_growth_by_source_opt_sigma.pdf", format="pdf")
"""

#plot countries in pool
x=np.arange(2)
plt.figure(3)
w_p= np.zeros((2,5))
for i in range(0,5):
    w_p[1,i]=np.mean(w_g[0:n_country-1,i])
w_p[0,:]=w_g[n_country-1,:]

print("estimated contributions K/L, TFP, Labor, Observed, Error\n",w_p)

plt.bar(x-1.5*w, w_p[:,2],w, color='#4285f4')
plt.bar(x -0.5*w, w_p[:,1],w,color='#ea4335')
plt.bar(x+0.5*w, w_p[:,0],w, color='#fbbc04')
plt.bar(x+1.5*w, w_p[:,3],w, color="green")
if error_flag: plt.bar(x+2.5*w, w_p[:,4],w, color="purple")
plt.xticks(x, ["US","Others"])
plt.legend(["Labor share", "TFP","K/L","Observed Total","Error"], loc="upper right")
plt.ylabel("Mean Annual Real Wage Growth (%)", fontsize=14)
plt.savefig("series_wage_growth_by_source_pool_sigma.pdf", format="pdf")


#compute mean labor share change over span for all countries
dlogL_dt=np.zeros(n_country)
for c in range(0,n_country): dlogL_dt[c]=np.log(ls[c_end[c]-1]/ls[c_start[c]])/(c_end[c]-c_start[c])

print("US err%, Annual LS_change% LS_contrib %",100*w_p[0,4]/w_p[0,3],
      100*dlogL_dt[n_country-1],100*w_p[0,2]/w_p[0,3])
print("others err%, Annual LS_change% LS_contrib %",100*w_p[1,4]/w_p[1,3],
      100*np.mean(dlogL_dt[0:n_country-1]), 100*w_p[1,2]/w_p[1,3])




"""
# plots of labor share
plt.figure(8)
for c in range(0,n_country):
    c_s=c_start[c];c_e=c_end[c]
    plt.plot(np.arange(0,c_e-c_s),xdata[c_s:c_e,5])
plt.title("Labor Share")
"""




plt.show()

















