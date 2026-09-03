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
country_list=["IT","US"]
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

# observed and predicted wages for comparision
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

beta_pool = np.zeros(3)
beta_pool[1] = pool_model.params[0]
beta_pool[0] = beta_pool[1] - 1.96*pool_model.bse[0]
beta_pool[2] = beta_pool[1] + 1.96*pool_model.bse[0]
sigma_pool = 1/(1 + beta_pool)
print("beta_pool",beta_pool, pool_model.bse[0])
print("sigma_pool, SE",sigma_pool, max(np.abs(sigma_pool-sigma_pool[1])))

print("\n\n\n")

beta_us = np.zeros(3)
beta_us[1] = us_model.params[0]
beta_us[0] = beta_us[1] - 1.96*us_model.bse[0]
beta_us[2] = beta_us[1] + 1.96*us_model.bse[0]
sigma_us = 1/(1 + beta_us)
print("beta_us, SE",beta_us, us_model.bse[0])
print("sigma_us, SE",sigma_us, max(np.abs(sigma_us-sigma_us[1])))


#estimate mean derivatives dln(w)/dln(lambda)
derivs=np.zeros((n_country,3,3))     # (country, low/mean/high, direct/indirect/net)
err_derivs=np.zeros(n_country)
for c in range(0,n_country):
    c_s=c_start[c];c_e =c_end[c]
    for i in range(3):
        if c<n_country-1:
            beta = beta_pool[i]
        else:
            beta = beta_us[i]
        derivs[c,i,0] = 1      # direct effect
        derivs[c,i,1] = np.mean(  #indirect effect
            - ls[c_s:c_e]*np.log(k[c_s:c_e])*
            (1 - 0.5*beta*(1-2*ls[c_s:c_e])*np.log(k[c_s:c_e])
            ))
        derivs[c,i,2] = derivs[c,i,0]+derivs[c,i,1]   #net effect

    err_derivs[c] = max(abs(derivs[c,1,2]-derivs[c,0,2]), abs(derivs[c,1,2]-derivs[c,2,2])) #worst_case SE


deriv_avg = derivs[:,1,:]



print("\n\nnegative of dln(w)/dln(lambda)\n",-deriv_avg)
print("standard error", err_derivs)


order =np.argsort(-deriv_avg[:,2]) # plot in order of increasing net derivative
c_list=[country_list[i] for i in order]
fig, ax = plt.subplots()
ax.axhline(0.0,color='gray', linewidth=0.5)
m_size= 10 # marker size

plt.scatter(c_list, -deriv_avg[order,0], label="Direct",c='#4285f4',s=m_size)

plt.scatter(c_list, -deriv_avg[order,1] ,label="Indirect",c='#ea4335',s=m_size,marker='s')

plt.scatter(c_list, -deriv_avg[order,2], label="Net", c='#fbbc04',s=m_size,marker='D')
plt.errorbar(c_list, -deriv_avg[order,2],yerr=err_derivs[order],fmt='none',ecolor='#fbbc04')

plt.legend(loc="upper left")
plt.ylabel("Derivative",fontsize=14)
plt.savefig("series_wage_derivative.pdf", format="pdf")

"""
plt.figure(4)
plt.plot(np.arange(0,N),log_w)
plt.plot(np.arange(0,N),pred_log_w)
"""




#This section plots wage growth contributions by source
w_g=np.zeros((n_country,3, 5)) # (country, type=(min, mean, max), (K/L,TFP, labor_share, observed_total, error))

for c in range(0,n_country):
    c_s=c_start[c]; c_e=c_end[c]
    for i in range(3): #min, mean, max
        if c<n_country-1:
            beta = beta_pool[i]
        else:
            beta = beta_us[i]

        #TFP contribution to wage growth
        w_g[c,i,1] = (np.log(xdata[c_e-1,1]/xdata[c_s,1]))/(c_e-c_s)


        #k contribution as mean of forward and backward differences
        w_g[c,i,0] = np.mean(
                    0.5*(np.log(k[c_s+1:c_e]/k[c_s:c_e-1]))*
                    (1-ls[c_s:c_e-1])*(1 + beta*ls[c_s:c_e-1]*np.log(k[c_s:c_e-1])) +
                    0.5*(np.log(k[c_s+1:c_e]/k[c_s:c_e-1]))*
                    (1-ls[c_s+1:c_e])*(1 + beta*ls[c_s+1:c_e]*np.log(k[c_s+1:c_e]))
                            )


        #labor share contribution as mean of forward and backward difference
        w_g[c,i,2]= np.mean(
                0.5*(np.log(ls[c_s+1:c_e]/ls[c_s:c_e-1]))*(1 -
                    ls[c_s:c_e-1]*np.log(k[c_s:c_e-1])* (1 - 0.5*beta*(1 - 2*ls[c_s:c_e-1])*np.log(k[c_s:c_e-1]))) +
                0.5*(np.log(ls[c_s+1:c_e]/ls[c_s:c_e-1]))*(1 -
                ls[c_s+1:c_e]*np.log(k[c_s+1:c_e])* (1 - 0.5*beta*(1 - 2*ls[c_s+1:c_e])*np.log(k[c_s+1:c_e]))))

        # calculate from wage deriv to verify
        #w_g[c,i,2] = deriv_avg[c,2]*(np.log(ls[c_e-1])-np.log(ls[c_s]))/(c_e-c_s)

        #observed total wage growth
        w_g[c,i,3] = (log_w[c_e-1] - log_w[c_s])/(c_e-c_s)

        # error between observed total and estimated wage growth
        w_g[c,i,4] = np.sum(w_g[c,i,0:3]) - w_g[c,i,3]


w_g *=100 #convert to percentages
if error_flag: w = 0.15 # bar width and spacing
else: w=0.2

#plot countries in pool
x=np.arange(2)
plt.figure(3)
w_p = np.zeros((2,3,5))  #(country=(US, pool), type=(min, mean, max), (K/L,TFP, labor_share, observed_total, error))
w_p[1] = np.mean(w_g[0:n_country-1], axis=0) #pool
w_p[0] = w_g[n_country-1] #US
err_w_p = np.array([w_p[:,1] - w_p[:,0], w_p[:,2] - w_p[:,1]])   #upper and lower 95% confidence envelopes

print("estimated contributions K/L, TFP, Labor, Observed, Error\n",w_p[:,1])
print("1.96*SE confidence band of wage-growth contributions K/L, TFP, Labor, Observed, Error\n",err_w_p)

plt.bar(x-1.5*w, w_p[:,1,2],w, color='#4285f4', yerr=err_w_p[...,2], capsize=5)
plt.bar(x -0.5*w, w_p[:,1,1],w,color='#ea4335', yerr=err_w_p[...,1], capsize=5)
plt.bar(x+0.5*w, w_p[:,1,0],w, color='#fbbc04', yerr=err_w_p[...,0], capsize=5)
plt.bar(x+1.5*w, w_p[:,1,3],w, color="green")
#if error_flag: plt.bar(x+2.5*w, w_p[:,1, 4],w, color="purple", yerr=err_w_p[...,4], capsize=5)
if error_flag: plt.bar(x+2.5*w, w_p[:,1, 4],w, color="purple")
plt.xticks(x, ["US","Others"])
plt.legend(["Labor share", "TFP","K/L","Observed Total","Error"], loc="upper right")
plt.ylabel("Mean Annual Real Wage Growth (%)", fontsize=14)
plt.savefig("series_wage_growth_by_source_pool_sigma.pdf", format="pdf")


#compute mean labor share change over span for all countries
dlogL_dt=np.zeros(n_country)
for c in range(0,n_country): dlogL_dt[c]=np.log(ls[c_end[c]-1]/ls[c_start[c]])/(c_end[c]-c_start[c])

print("US err%, Annual LS_change% LS_contrib %",100*w_p[0,1,4]/w_p[0,1,3],
      100*dlogL_dt[n_country-1],100*w_p[0,1,2]/w_p[0,1,3])
print("others err%, Annual LS_change% LS_contrib %",100*w_p[1,1,4]/w_p[1,1,3],
      100*np.mean(dlogL_dt[0:n_country-1]), 100*w_p[1,1,2]/w_p[1,1,3])




"""
# plots of labor share
plt.figure(8)
for c in range(0,n_country):
    c_s=c_start[c];c_e=c_end[c]
    plt.plot(np.arange(0,c_e-c_s),xdata[c_s:c_e,5])
plt.title("Labor Share")
"""

####################################################################
#update of dec 27 2025 to calculate wages at sigma_star at latest point in data series
####################################################################

max_wage = np.zeros(n_country)
lambda_star = np.zeros(n_country)
lambda_cur = np.zeros(n_country)
err_lambda_star = np.zeros((2,n_country))
for c in range(n_country):
    c_s=c_start[c]; c_e=c_end[c]; c_s = c_e-1 #most-recent year; can be any span of the series
    log_k = np.log(k[c_s:c_e])
    if c == n_country -1:
        beta_min = beta_us[0]
        beta = beta_us[1]
        beta_max = beta_us[2]
    else:
        beta_min = beta_pool[0]
        beta = beta_pool[1]
        beta_max = beta_pool[2]
    l_star = np.mean((beta*log_k - 2 + np.sqrt((beta*log_k - 2)**2 +16*beta))/(4*beta*log_k)) # positive root of quadratic
    l_star_max = np.mean((beta_max*log_k - 2 + np.sqrt((beta_max*log_k - 2)**2 +16*beta_max))/(4*beta_max*log_k))
    l_star_min = np.mean((beta_min*log_k - 2 + np.sqrt((beta_min*log_k - 2)**2 +16*beta_min))/(4*beta_min*log_k))
    lambda_star[c] = l_star
    lambda_cur[c] = np.mean(xdata[c_s:c_e,5])
    #SE_lambda_star[c] = max(abs(l_star_max-l_star), abs(l_star_min-l_star))
    err_lambda_star[:,c] = [l_star - l_star_min, l_star_max - l_star]


plt.figure(5)
order=np.argsort(lambda_cur)
c_list=[country_list[i] for i in order]
x=np.arange(len(order))
plt.scatter(c_list, lambda_star[order], label="Wage-maximizing Labor Share",c='#4285f4',s=m_size)
plt.errorbar(c_list, lambda_star[order],yerr=err_lambda_star[:,order],fmt='none',ecolor='#4285f4')
plt.scatter(c_list, lambda_cur[order], label="Prevailing Labor Share",c='#ea4335',s=m_size,marker='s')
plt.ylabel("Labor Share (%)", fontsize=14)
plt.legend(loc="lower left")


plt.show()

















