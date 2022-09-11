import pandas as pd, math, numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm


y = 0.04
n = 6.0
m = 0.5
sigma = 0.1
Asset_Price = 10000
strike = 10000

h = m / n
delta = -y

u = math.exp(sigma * (h**0.5))
d = math.exp(-sigma * (h**0.5))
p = (math.exp(y*h)-d)/(u-d)

print "### EXERCISE 6.1 ###"
print "Part A"
STOCK = pd.DataFrame(index = np.arange(7), columns=["Period","Asset_Price", "Period_0", "Period_1", "Period_2", "Period_3", "Period_4", "Period_5", "Period_6"])
STOCK["Asset_Price"][0] = Asset_Price
STOCK["Period_0"][0] = Asset_Price

for i in range(1,len(STOCK.columns.values)):
	STOCK["Asset_Price"][i] = math.exp(-y*h)*(p*STOCK["Asset_Price"][i-1]*u+(1-p)*d*STOCK["Asset_Price"][i-1])

for i in range(3,len(STOCK.columns.values)):
	STOCK["Period_" + str(i-2)][0:i-1] = 0
	for j in range(i-2):
		STOCK["Period_" + str(i-2)][j] = STOCK["Period_" + str(i-3)][j]*d
	STOCK["Period_" + str(i-2)][j+1] = STOCK["Period_" + str(i-3)][j]*u
print STOCK


print "Part B"
FUTURE = pd.DataFrame(index = np.arange(7), columns=["Period","Asset_Price", "Period_0", "Period_1", "Period_2", "Period_3", "Period_4", "Period_5", "Period_6"])
FUTURE["Asset_Price"][0] = STOCK["Asset_Price"][0]
FUTURE["Period_6"] = STOCK["Period_6"]

for j in range(len(FUTURE.columns.values)-2,1,-1):
	#print j
	for k in range(j-1):
		p = (math.exp((y-delta)*h)-d)/(u-d)
		FUTURE[FUTURE.columns.values[j]][k] = math.exp(-y*h)*((1-p)*FUTURE[FUTURE.columns.values[j+1]][k] + (p)*FUTURE[FUTURE.columns.values[j+1]][k+1])
print STOCK
print FUTURE


print "Part C"
OPTION = pd.DataFrame(index = np.arange(4), columns=["Period","Asset_Price", "Period_0", "Period_1", "Period_2", "Period_3"])
for i in range(len(OPTION.columns.values)-1,1,-1):
	for j in range(0,i-1):
		if i == len(OPTION.columns.values)-1:
			OPTION[OPTION.columns.values[i]][j] = max([strike - STOCK[STOCK.columns.values[i]][j],0])
		else:
			OPTION[OPTION.columns.values[i]][j] = max([strike-STOCK[STOCK.columns.values[i]][j], math.exp(-y*h)*((1-p)*max([strike-STOCK[STOCK.columns.values[i+1]][j] ,0]) + (p)*max([strike-STOCK[STOCK.columns.values[i+1]][j+1] ,0]) )])
		#print "	" + str(j)
		#OPTION[OPTION.columns.values[i]][j] = max(list([strike - ]))


print OPTION

print "### EXERCISE 6.2 ###"
print "Part A"

S = 10000
K = 10000
T = 3.0/12.0
r = 0.04
d = 0.04
V = 0.1

d1 = (math.log(float(S)/K) + ((r-d)+ V**2.0/2.0)*T)/(V*math.sqrt(T))
d2 = d1-V*math.sqrt(T)

C_sigma = K*math.exp(-r*T)*norm.cdf(-d2)-S*math.exp(-d*T)*norm.cdf(-d1)
print "BS Calculated:	" + str(round(C_sigma,2))

print "Part B-C"
Implied_Vol = []
BS_Price = []
differences =[0.80, 0.90, 1.10 ,1.20]
for difference in differences:
	C_sigma_observed = difference * C_sigma
	print "BS Observed:	" + str(round(C_sigma_observed,2))
	O = 1000
	Vol = 100.0
	for i in range(10,200,1):
		V = round(i / 1000.0,5)
		d1 = (math.log(float(S)/K) + ((r-d)+ V**2.0/2.0)*T)/(V*math.sqrt(T))
		d2 = d1-V*math.sqrt(T)
		Price = K*math.exp(-r*T)*norm.cdf(-d2)-S*math.exp(-d*T)*norm.cdf(-d1)
		diff = abs(Price - C_sigma_observed)
		if diff < O:
			O = diff
			BSPrice = Price
			Vol = V

	Implied_Vol.append(Vol)
	BS_Price.append(BSPrice)
	print "Implied Volatility for BS price:	" + str(C_sigma_observed) + "	is	" + str(Vol)	


plt.plot(Implied_Vol, BS_Price, 'o-')
plt.title('Relation between Implied Volatility & Option Price', fontsize=15)
plt.xlabel('Implied Volatility', fontsize=12)
plt.ylabel('Option Price', fontsize=12)
plt.xlim(0.07,0.13)
plt.ylim(150,250)
plt.show()




