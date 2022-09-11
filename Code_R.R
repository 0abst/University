#University of Sankt Gallen, Switzerland

#Master thesis

################################################
########## Code R - 12 August 2016##############
################################################

#########################
#Packages to be installed
#########################

install.packages("xlsx")
install.packages("foreign") #for using Stata files
install.packages("sde") #for the GBM package R
install.packages("aod")
install.packages("relaimpo") #relative importance linear model predictors
install.packages("lmtest")
install.packages("texreg")  # Tex Printing
install.packages("xtable")  # Tex Printing
install.packages("DAMisc")
install.packages("Hmisc") #nedded for rcorr()
install.packages("popbio") #plotting logistic regression
install.packages("effects")
install.packages("lattice")
install.packages("AER") #Kleiber/Zeileis. 2008. Applied Econometrics with R. Springer
install.packages("pspline")
install.packages("splines")
install.packages("mgcv")
install.packages("fBasics")
install.packages("asympTest") #dispersion tests ("mood, ansari, siegel tukey etc")
install.packages("exactRankTests") #dispersion tests ("mood, ansari, siegel tukey etc")
install.packages("rattle") #data mining
install.packages("gmodels") #crosstabs
install.packages("asbio") #nice correlograms
install.packages("corrgram") #nice correlograms
install.packages("memisc") #data management
install.packages("polynom") #for loading polynoms
install.packages("vcd") # #mosaic plots+ association tests (contingency coeff etc
install.packages("sm") #plotting kernel density of groups
install.packages("quantreg") #quantile regression
install.packages("rJava") #quantile regression
install.packages("openxlsx") #quantile regression
install.packages("corrgram") #for the graphs of the correlation matrix
install.packages("ggplot2") #for the graphs of the correlation matrix
install.packages("rgl")
install.packages("matrixcalc") #test if the matrix is positive definite
install.packages("fOptions") #for making prices paths
install.packages("gdata") #Library gdata(for reading Excel files)
install.packages("stats4")
install.packages("bbmle") #tools for the mle
install.packages("yuima") #tools for the mle
install.packages("sfsmisc")
install.packages("corpcor")
install.packages("EnvStats") #for applying the log normal test
install.packages("xts") #packages for time series
install.packages("timeSeries") #packages for time series
install.packages("dynlm") #for regressions with time series
install.packages("Metrics") #for the measures of goodness of fit
install.packages("LICORS") #for introducing thresholds
install.packages("corrplot") #for correlograms
install.packages("corrgram") #for correlograms
install.packages("xtable") #for exporting data to Latex
install.packages("LICORS")
install.packages("Sim.DiffProc") #for testing the lognormal distribution
install.packages("knitr") #for exporting the pdf

#########################
#Libraries to be loaded
#########################

library(xlsx)
library(foreign) #for using Stata files
library(sde) #for the GBM package R
library(aod)
library(relaimpo) #relative importance linear model predictors
library(lmtest)
library(texreg)  # Tex Printing
library(xtable)  # Tex Printing
library(DAMisc)
library(Hmisc) #nedded for rcorr()
library(popbio) #plotting logistic regression
library(effects)
library(lattice)
library(AER) #Kleiber/Zeileis. 2008. Applied Econometrics with R. Springer
library(pspline)
library(splines)
library(mgcv)
library(fBasics)
library(asympTest) #dispersion tests (mood, ansari, siegel tukey etc)
library(exactRankTests) #dispersion tests (mood, ansari, siegel tukey etc)
library(rattle) #data mining
library(gmodels) #crosstabs
library(asbio) #nice correlograms
library(corrgram) #nice correlograms
library(memisc) #data management
library(polynom) #for loading polynoms
library(vcd) # #mosaic plots+ association tests (contingency coeff etc
library(sm) #plotting kernel density of groups
library(quantreg) #quantile regression
library(rJava) #quantile regression
library(openxlsx) #quantile regression
library(corrgram) #for the graphs of the correlation matrix
library(ggplot2) #for the graphs of the correlation matrix
library(rgl)
library(matrixcalc) #test if the matrix is positive definite
library(fOptions) #for making prices paths
library(gdata) #Library gdata(for reading Excel files)
library(stats4) #for the mle
library(bbmle) #tools for the mle
library(yuima) #tools for the qmle
library(corpcor) #for having a positive definite matrix
library(EnvStats) #for applying the log normal test
library(xts) #packages for time series
library(timeSeries) #packages for time series
library(dynlm) #for regressions with time series
library(Metrics) #for the measures of goodness of fit
library(LICORS) #for introducing thresholds
library(corrplot) #for correlograms
library(corrgram) #for correlograms
library(xtable) #for exporting data to Latex
library(LICORS)
library(Sim.DiffProc) #for testing log normal distribution
library(knitr) #for exporting the pdf

#########################
####PLAN OF THE CODE
#########################

#I. INTEREST RATES AND CAP RATES ARE TAKEN AS "CONSTANT"

#II. OLS/ MLE ESTIMATIONs FOR THE CAP RATES PARAMETERS

#III. OLS/ MLE ESTIMATION FOR THE INTEREST RATES PARAMETERS

#IV. MEASURES OF GOODNESS OF FIT/ KOLMOGOROV SMIRNOV TESTS

#EXTRACTION OF THE PARAMETERS/ EXPORTATION TO LATEX

#################################################
####SOURCES OF DOCUMENTS USED FOR MAKING THE CODE
#################################################

#The following main sources have been used for making the code.
##http://stackoverflow.com/questions/12480034/using-lm-in-r-for-a-series-of-independent-fits
#http://stackoverflow.com/questions/18067519/using-r-to-do-a-regression-with-multiple-dependent-and-multiple-independent-vari
#we apply an algorithm to make it positive definite
#https://stat.ethz.ch/R-manual/R-devel/library/Matrix/html/nearPD.html                                       
#For the graphs:
#http://stackoverflow.com/questions/11949331/adding-a-3rd-order-polynomial-and-its-equation-to-a-ggplot-in-r
#http://stackoverflow.com/questions/10438752/adding-x-and-y-axis-labels-in-ggplot2
#Data for the Risk free rate: https://research.stlouisfed.org/fred2/series/DTB3
#Paper of reference for the thesis:
#Goldbeck and Linetsky, Least Squares Monte Carlo Valuation of Residential Mortgages with Prepayment and Default Risk, available at the following URL:
#http://faculty.washington.edu/golbeck/MortgageSim.pdf
#for the regression with time series for the OLS
#http://stats.stackexchange.com/questions/92498/forecasting-time-series-regression-in-r-using-lm
#For the MLE estimation:
#"Simulation and Inference for Stochastic Differential Equations by Stefano M. Iacus
#Springer Series in Statistics, 2008
#For the correlograms we used http://www.statmethods.net/advgraphs/correlograms.html
#Modeling and simulating Interest Rates via Time-Dependent Mean Reversion
#by Andrew Jason Dweck (2014) - pages 15-19

#In the code, when sources are used, references are added.

##########################################
####PATH TO BE CHANGED BY THE USER
##########################################

#the Excel file of the Monte Carlo simulations shall be available in the directory set. 
Dir="C:/Users/Moi/Documents/HSG/Master thesis Real Estate/Codes/California data/We separate the properties by sub types" 
setwd(Dir)

############################################################
#I. INTEREST RATES AND CAP RATES ARE TAKEN AS "CONSTANT"
############################################################

#We load the data from the Excel file

#warning: we just select the columns we are going to work with (1:9)
#this extraction gives most of the data, with the exception of the data used to compute the covariance matrix
data_apartment_individual_property=read.xlsx("Classeur_Excel_tri_data_MC_end_value_43.xlsm", sheet = "Apartment_final_elements_VBA", startRow = 1, colNames = TRUE,
                                             rowNames = FALSE, detectDates = FALSE, skipEmptyRows = TRUE,
                                             rows = NULL, cols = 1:9, check.names = FALSE, namedRegion = NULL)                          


#The present extraction gives the data which is going to be used to compute the covariance matrix
data_apartment_matrix_cap_rates=read.xlsx("Classeur_Excel_tri_data_MC_end_value_43.xlsm", sheet = "Cap_rates_Apartment_Final_Mtx", startRow = 1, colNames = TRUE,
                                          rowNames = TRUE, detectDates = FALSE, skipEmptyRows = TRUE,
                                          rows =c(1:18), cols = NULL, check.names = FALSE, namedRegion = NULL)

#Loading the function Asset Paths
#This function is used for computing the values of the different properties at various point in time
#This function needs to be loaded for computing the prices
#It has been taken from http://www.r-bloggers.com/simulating-multiple-asset-paths-in-r/
#and written by Michael Kapler in 2012 (this date has been estimated)

asset.paths <- function(s0, mu, sigma, 
                        nsims = 100, 
                        periods = c(0, 1)   # time periods at which to simulate prices
) 
{
  s0 = as.vector(s0)
  nsteps = length(periods)
  dt = c(periods[1], diff(periods))
  
  if( length(s0) == 1 ) {
    drift = mu - 0.5 * sigma^2
    if( nsteps == 1 ) {
      s0 * exp(drift * dt + sigma * sqrt(dt) * rnorm(nsims))
    } else {
      temp = matrix(exp(drift * dt + sigma * sqrt(dt) * rnorm(nsteps * nsims)), nc=nsims)
      for(i in 2:nsteps) temp[i,] = temp[i,] * temp[(i-1),]
      s0 * temp
    }
  } else {
    require(MASS)
    drift = mu - 0.5 * diag(sigma)
    n = length(mu)
    
    if( nsteps == 1 ) {
      
      s0 * exp(drift * dt + sqrt(dt) * t(mvrnorm(nsims, rep(0, n), sigma)))
    } else {
      temp = array(exp(as.vector(drift %*% t(dt)) + t(sqrt(dt) * mvrnorm(nsteps * nsims, rep(0, n), sigma))), c(n, nsteps, nsims))
      for(i in 2:nsteps) temp[,i,] = temp[,i,] * temp[,(i-1),]
      s0 * temp
    }
  }
}

##########################################
####COMPUTATION OF THE COVARIANCE MATRIX
##########################################

#If we compute directly the covariance matrix, it is not positive definite.
#However, we would like to have a positive definite matrix.
#For this purpose, we use the package corpcor.
data_apartment_matrix_cap_rates_matrix=as.matrix(data_apartment_matrix_cap_rates)
#we compute the covariance.
cov_data_apartment_matrix_cap_rates_matrix=cov(data_apartment_matrix_cap_rates_matrix)
#we apply the command "make positive definite". This command uses the "corpcor" package.
make_positive_cov_data_apartment_matrix_cap_rates_matrix=make.positive.definite(cov_data_apartment_matrix_cap_rates_matrix)
is.positive.definite(make_positive_cov_data_apartment_matrix_cap_rates_matrix)

#The covariance matrix is now:
covariance_matrix_apartment=make_positive_cov_data_apartment_matrix_cap_rates_matrix

#command used to produce the correlation
correlation_matrix_apartment=cor(data_apartment_matrix_cap_rates)

#Correlogram
#code for producing the correlogram has been taken from: http://www.statmethods.net/advgraphs/correlograms.html
corrgram_apartment=corrgram(data_apartment_matrix_cap_rates, order=NULL, lower.panel=panel.shade,
         upper.panel=NULL, text.panel=panel.txt,
         main="Correlogram of the cap rates - Apartment")

################################################
####LOADING VECTORS TO BE USED IN THE SIMULATION
################################################

#Intial Acquisition cost
apartment_Init_acq_cost=data_apartment_individual_property$Init_acq_cost_col
#Net Sale Price
apartment_NetSalePrice=data_apartment_individual_property$NetSalePrice_col
#we display a graph to see if the distribution is log normal
x <-apartment_NetSalePrice
range_x <- range(x)
seq_log_norm=seq(0, range_x[2], by=10000)
d <- dlnorm(seq_log_norm, meanlog = mean(log(x)), sdlog = sd(log(x)))
#hist(x, prob = TRUE, ylim = range(d),main="Distribution of the Net Sales Prices for the Property Type Apartment",xlab = "Values of the Net Sales Prices")
hist_x=hist(x,prob = TRUE, main="Distribution of the Net Sales Prices for the Property Type Apartment",xlab = "Values of the Net Sales Prices")
lines(density(x),col="blue")
lines(seq_log_norm, d, col="red")
legend("topright", legend=c("Histogram of the Net Sales Prices","Density of the Net Sales Prices","Fitted log normal density of the Net Sales Prices"), lty=2, cex=0.8, pch=c(1,19),col=c("black","blue","red"))

#test for a log normal distribution
mean_log_x_apartment <- mean(log(x)) 
sd_log_x_apartment <- sd(log(x)) 
test_net_sales_prices_log_normal_dist_apartment=ks.test(x,"plnorm",mean_log_x_apartment,sd_log_x_apartment) 
test_net_sales_prices_log_normal_dist_apartment_statistic=test_net_sales_prices_log_normal_dist_apartment$statistic
test_net_sales_prices_log_normal_dist_apartment_p_value=test_net_sales_prices_log_normal_dist_apartment$p.value
test_net_sales_prices_log_normal_dist_apartment_statistic_p_value=c(test_net_sales_prices_log_normal_dist_apartment_statistic,test_net_sales_prices_log_normal_dist_apartment_p_value)
 
#we load the time vector of the initial time
apartment_year_built_col=data_apartment_individual_property$Year_built_col
#we load the vector for the final time - this is the year which corresponds to the out of sample predictions
apartment_year_col=data_apartment_individual_property$Year_col
#we load the vector which gives the time difference
apartment_time_difference=data_apartment_individual_property$apartment_time_difference

################################################
####COMPUTATION OF RETURNS TO BE USED IN THE GBM
################################################

#Returns
#we consider different methods when using interest rates
#Case 1: same interest rate for every property
#Case 2: take the one adapted to the property but still constant
#Case 3: we model the Cap rates with a Vasicek model and the interest rates through a Cox process (this part is later)

#For the cases 1 and 2, for the equivalent of the dividends, we take the average of the non zero cap rates from the first value available to
#the one before the last, as the price associated will be predicted and hence it would not be an out-of-sample estimation but a in-sample estimation.
#We load the dividends - the cap rates are considered dividends in our model
apartment_dividends=data_apartment_individual_property$CapRate_col

#interest rates - 
#Case 1: we fix it at 5%
apartment_risk_free_rate=rep(0.05,length(apartment_dividends))
#Case 2:
#what are we doing? We take the average of the risk free interest rates for every period and for each property:
#that means that for instance for an estimation which lasts from 1950 to 2012, we will take the average of the interest
#rates between 1950 and 2012. The code for this part is available in the Vba code associated to the Excel file.
#the risk free rates considered are the 3 months maturity T Bill from the FED.
#The data is available here: https://research.stlouisfed.org/fred2/series/DTB3

#warning: in the Excel file the interest rates are in percentages - this is why we divide them here by 100
apartment_Risk_free_rate_3_m_adapted=data_apartment_individual_property$Risk_free_rate_3_m_adapted/100

#Returns: we need to take the interest rates and to substract the dividends from the interest rates
#case1:
apartment_returns1=apartment_risk_free_rate-apartment_dividends

#case2:
apartment_returns2=apartment_Risk_free_rate_3_m_adapted-apartment_dividends

##we compute now the prices of each asset
#case1:
prices_apartment1=asset.paths(apartment_Init_acq_cost, apartment_returns1, covariance_matrix_apartment, 100, periods=apartment_time_difference)

#case2:
prices_apartment2=asset.paths(apartment_Init_acq_cost, apartment_returns2, covariance_matrix_apartment, 100, periods=apartment_time_difference)

#we compute the average price for each asset after simulation - we need to take the price at the end of the simulation - 
average_asset_price_apartment1=rep(0,length(apartment_dividends))
average_asset_price_apartment2=rep(0,length(apartment_dividends))

##We take the average price in the end of the time period
for (i in 1:length(apartment_dividends)){
  average_asset_price_apartment1[i]=mean(prices_apartment1[i,length(apartment_time_difference),])
  average_asset_price_apartment2[i]=mean(prices_apartment2[i,length(apartment_time_difference),])
}

#we create a vector in which there is a price comparison between the actual prices and the prices simulated
prices_comparison_apartment1=apartment_NetSalePrice-average_asset_price_apartment1
prices_comparison_apartment2=apartment_NetSalePrice-average_asset_price_apartment2

#we load a vector for which we have the estimation of the property through the hedonic pricing method
#this gives us a benchmark to be used
prices_estimation_MVLag1_apartment=data_apartment_individual_property$MVLag1_apartment

#we compute the benchmark - the prices comparison between the net sales prices and the estimation of the property through the hedonic pricing method
prices_comparison_benchmark_apartment=apartment_NetSalePrice-prices_estimation_MVLag1_apartment
#for vizualizing the data
hist(prices_estimation_MVLag1_apartment,main="Distribution of the Market values lagged by one quarter apartment",xlab = "Observations")

#elements for the graphs
data_for_graph_benchmark_apartment=cbind(1:length(prices_comparison_benchmark_apartment),prices_comparison_benchmark_apartment)
data_for_graph_benchmark_apartment=as.data.frame(data_for_graph_benchmark_apartment)

################################################################
#GRAPHS FOR INTEREST RATES AND CAP RATES ARE TAKEN AS "CONSTANT"
###############################################################

#then we draw graphs to compare the actual prices and the prices obtained after Monte-Carlo simulations
#the present application of the package ggplot2 has been taken from:
#http://stackoverflow.com/questions/11949331/adding-a-3rd-order-polynomial-and-its-equation-to-a-ggplot-in-r
#the idea for making two graphs has been taken from:
#http://stackoverflow.com/questions/21192002/how-to-combine-2-plots-ggplot-into-one-plot
#the following url has also been used:
#http://stackoverflow.com/questions/10438752/adding-x-and-y-axis-labels-in-ggplot2
data_for_graph1=cbind(1:length(prices_comparison_apartment1),prices_comparison_apartment1)
data_for_graph2=cbind(1:length(prices_comparison_apartment2),prices_comparison_apartment2)

#we convert the data into a data frame - this is needed for a good application of the graphs
data_for_graph1=as.data.frame(data_for_graph1)
data_for_graph2=as.data.frame(data_for_graph2)

#########
#Graph 1 - Prices Comparison when the cap rates are constant and directly loaded from the Excel Sheet and the interest rates are fixed at 5%
#########

###############################
###We keep the outliers
###############################

prices_comparison_apartment1_with_outlier <- prices_comparison_apartment1[prices_comparison_apartment1 > -1e+200] 
prices_comparison_apartment1_with_outlier <- prices_comparison_apartment1_with_outlier[!prices_comparison_apartment1_with_outlier > 1e+200] 

indices_outliers_graph1_apartment_condition_1=which(prices_comparison_apartment1 < -1e+200)
indices_outliers_graph1_apartment_condition_2=which(prices_comparison_apartment1 > 1e+200)

#we create a vector which gathers the elements which shall be removed according to the conditions specified above (outliers)
indices_outliers_graph1_apartment=c(indices_outliers_graph1_apartment_condition_1,indices_outliers_graph1_apartment_condition_2)

data_for_graph1_with_outlier=cbind(1:length(prices_comparison_apartment1_with_outlier),prices_comparison_apartment1_with_outlier)
data_for_graph1_with_outlier=as.data.frame(data_for_graph1_with_outlier)

x <- data.frame(x = 1:length(prices_comparison_apartment1_with_outlier)) 
df1 <- data.frame("x"=x, "y1"=data_for_graph1_with_outlier$prices_comparison_apartment1_with_outlier)

#we need to change the benchmark - we remove the values which correspond to the indices of the outliers
prices_comparison_benchmark_apartment_graph_1_with_outlier=prices_comparison_benchmark_apartment
#length(prices_comparison_benchmark_apartment_graph_3)
data_for_graph_1_with_outlier_benchmark=cbind(1:length(prices_comparison_apartment1_with_outlier),prices_comparison_benchmark_apartment_graph_1_with_outlier)
data_for_graph_1_with_outlier_benchmark=as.data.frame(data_for_graph_1_with_outlier_benchmark)

df2 <- data.frame("x"=x, "y2"=data_for_graph_1_with_outlier_benchmark$prices_comparison_benchmark_apartment_graph_1_with_outlier)

my.formula1 <- df1$y1 ~ poly(x, 2, raw = TRUE)
my.formula2 <- df2$y2 ~ poly(x, 2, raw = TRUE)

#my.formula2
#p <- ggplot(df, aes(x, y))
p <- ggplot()+
  #four next lines of the code adapted from http://stackoverflow.com/questions/10438752/adding-x-and-y-axis-labels-in-ggplot2
  scale_size_area() + 
  xlab("Observations") +
  ylab("Prices difference") +
  ggtitle("Prices difference for Property type apartment with identical Interest Rates fixed at 5% - comparison with a benchmark - with outliers")+
  geom_point(data=df1,aes(x=x,y=df1$y1),alpha=2/10, shape=21, fill="blue", colour="black", size=5) +
  geom_point(data=df2,aes(x=x,y=df2$y2),alpha=2/10, shape=21, fill="green", colour="black", size=5) +
  geom_smooth(data=df1,aes(x=x,y=df1$y1),method = "lm", se = FALSE, formula = my.formula1, colour = "red") +
  geom_smooth(data=df2,aes(x=x,y=df2$y2),method = "lm", se = FALSE, formula = my.formula2, colour = "blue")

m1 <- lm(my.formula1, df1)
m2 <- lm(my.formula2, df2)

my.eq1 <- as.character(signif(as.polynomial(coef(m1)), 3))
my.eq2 <- as.character(signif(as.polynomial(coef(m2)), 3))
label.text1 <- paste(gsub("x", "~italic(x)", my.eq1, fixed = TRUE))
label.text2 <- paste(gsub("x", "~italic(x)", my.eq2, fixed = TRUE))

p + annotate(geom = "text", x = -Inf, y =max(prices_comparison_apartment1_with_outlier)/2 , label = label.text1,
             family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "red")+
  #             family = "serif", hjust = 0, parse = TRUE, size = 4)+
  
  annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment1_with_outlier)/4, label = label.text2, 
           family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "blue")


###############################
###We remove the outliers
###############################


#we create a vector "prices comparison apartment 2 without outliers"
prices_comparison_apartment1_without_outlier <- prices_comparison_apartment1[prices_comparison_apartment1 > -1e+3] 
prices_comparison_apartment1_without_outlier <- prices_comparison_apartment1_without_outlier[!prices_comparison_apartment1_without_outlier > 1e+8] 

#number of outliers removed
nb_outliers_removed_OLS_estimation_apartment_prices_comparison_apartment1=length(prices_comparison_apartment1)-length(prices_comparison_apartment1_without_outlier)

indices_outliers_graph1_apartment_condition_1=which(prices_comparison_apartment1 < -1e+3)
indices_outliers_graph1_apartment_condition_2=which(prices_comparison_apartment1 > 1e+8)

#we create a vector which gathers the elements which shall be removed according to the conditions specified above (outliers)
indices_outliers_graph1_apartment=c(indices_outliers_graph1_apartment_condition_1,indices_outliers_graph1_apartment_condition_2)

data_for_graph1_without_outlier=cbind(1:length(prices_comparison_apartment1_without_outlier),prices_comparison_apartment1_without_outlier)
data_for_graph1_without_outlier=as.data.frame(data_for_graph1_without_outlier)

x <- data.frame(x = 1:length(prices_comparison_apartment1_without_outlier)) 
df1 <- data.frame("x"=x, "y1"=data_for_graph1_without_outlier$prices_comparison_apartment1_without_outlier)

#we need to change the benchmark - we remove the values which correspond to the indices of the outliers
#we recreate a new dataframe adapted
#prices_comparison_benchmark_apartment
prices_comparison_benchmark_apartment_graph_1_without_outlier=prices_comparison_benchmark_apartment[-indices_outliers_graph1_apartment]
#length(prices_comparison_benchmark_apartment_graph_3)
data_for_graph_1_without_outlier_benchmark=cbind(1:length(prices_comparison_apartment1_without_outlier),prices_comparison_benchmark_apartment_graph_1_without_outlier)
data_for_graph_1_without_outlier_benchmark=as.data.frame(data_for_graph_1_without_outlier_benchmark)

#data_for_benchmark_graph3_apartment=prices_comparison_benchmark_apartment[prices_comparison_benchmark_apartment!=prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
#length(data_for_benchmark_graph3_apartment)
#data_for_benchmark_graph3_apartment=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[-data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
df2 <- data.frame("x"=x, "y2"=data_for_graph_1_without_outlier_benchmark$prices_comparison_benchmark_apartment_graph_1_without_outlier)

#df_benchmark <- data.frame("x"=x, "y"=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment)
my.formula1 <- df1$y1 ~ poly(x, 2, raw = TRUE)
my.formula2 <- df2$y2 ~ poly(x, 2, raw = TRUE)

#my.formula2
#p <- ggplot(df, aes(x, y))
p <- ggplot()+
  #four next lines of the code adapted from http://stackoverflow.com/questions/10438752/adding-x-and-y-axis-labels-in-ggplot2
  scale_size_area() + 
  xlab("Observations") +
  ylab("Prices difference") +
  ggtitle("Prices difference for Property type apartment with identical Interest Rates fixed at 5% - comparison with a benchmark - without outliers")+
  geom_point(data=df1,aes(x=x,y=df1$y1),alpha=2/10, shape=21, fill="blue", colour="black", size=5) +
  geom_point(data=df2,aes(x=x,y=df2$y2),alpha=2/10, shape=21, fill="green", colour="black", size=5) +
  geom_smooth(data=df1,aes(x=x,y=df1$y1),method = "lm", se = FALSE, formula = my.formula1, colour = "red") +
  geom_smooth(data=df2,aes(x=x,y=df2$y2),method = "lm", se = FALSE, formula = my.formula2, colour = "blue")

m1 <- lm(my.formula1, df1)
m2 <- lm(my.formula2, df2)

my.eq1 <- as.character(signif(as.polynomial(coef(m1)), 3))
my.eq2 <- as.character(signif(as.polynomial(coef(m2)), 3))
label.text1 <- paste(gsub("x", "~italic(x)", my.eq1, fixed = TRUE))
label.text2 <- paste(gsub("x", "~italic(x)", my.eq2, fixed = TRUE))

p + annotate(geom = "text", x = -Inf, y =max(prices_comparison_apartment1_without_outlier)/2 , label = label.text1,
             family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "red")+
  #             family = "serif", hjust = 0, parse = TRUE, size = 4)+
  
  annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment1_without_outlier)/4, label = label.text2, 
           family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "blue")


#########
#Graph 2 - Prices Comparison when the interest rates and the cap rates are constant and directly loaded from the Excel Sheet
########

###############################
###We keep the outliers
###############################

#we create a vector "prices comparison apartment 3 with outliers"
prices_comparison_apartment2_with_outlier <- prices_comparison_apartment2[prices_comparison_apartment2 > -1e+200] 
prices_comparison_apartment2_with_outlier <- prices_comparison_apartment2_with_outlier[!prices_comparison_apartment2_with_outlier > 1e+200] 

indices_outliers_graph2_apartment_condition_1=which(prices_comparison_apartment2 < -1e+200)
indices_outliers_graph2_apartment_condition_2=which(prices_comparison_apartment2 > 1e+200)

#we create a vector which gathers the elements which shall be removed according to the conditions specified above (outliers)
indices_outliers_graph2_apartment=c(indices_outliers_graph1_apartment_condition_1,indices_outliers_graph1_apartment_condition_2)

data_for_graph2_with_outlier=cbind(1:length(prices_comparison_apartment2_with_outlier),prices_comparison_apartment2_with_outlier)
data_for_graph2_with_outlier=as.data.frame(data_for_graph2_with_outlier)

x <- data.frame(x = 1:length(prices_comparison_apartment2_with_outlier)) 
df1 <- data.frame("x"=x, "y1"=data_for_graph2_with_outlier$prices_comparison_apartment2_with_outlier)

#we need to change the benchmark - we remove the values which correspond to the indices of the outliers
#we recreate a new dataframe adapted
#prices_comparison_benchmark_apartment
prices_comparison_benchmark_apartment_graph_2_with_outlier=prices_comparison_benchmark_apartment
#length(prices_comparison_benchmark_apartment_graph_3)
data_for_graph_2_with_outlier_benchmark=cbind(1:length(prices_comparison_apartment2_with_outlier),prices_comparison_benchmark_apartment_graph_2_with_outlier)
data_for_graph_2_with_outlier_benchmark=as.data.frame(data_for_graph_2_with_outlier_benchmark)

#data_for_benchmark_graph3_apartment=prices_comparison_benchmark_apartment[prices_comparison_benchmark_apartment!=prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
#length(data_for_benchmark_graph3_apartment)
#data_for_benchmark_graph3_apartment=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[-data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
df2 <- data.frame("x"=x, "y2"=data_for_graph_2_with_outlier_benchmark$prices_comparison_benchmark_apartment_graph_2_with_outlier)

#df_benchmark <- data.frame("x"=x, "y"=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment)
my.formula1 <- df1$y1 ~ poly(x, 2, raw = TRUE)
my.formula2 <- df2$y2 ~ poly(x, 2, raw = TRUE)

#my.formula2
#p <- ggplot(df, aes(x, y))
p <- ggplot()+
  #four next lines of the code adapted from http://stackoverflow.com/questions/10438752/adding-x-and-y-axis-labels-in-ggplot2
  scale_size_area() + 
  xlab("Observations") +
  ylab("Prices difference") +
  ggtitle("Prices difference for Property type apartment with adapted Interest Rates - with outliers")+
  geom_point(data=df1,aes(x=x,y=df1$y1),alpha=2/10, shape=21, fill="blue", colour="black", size=5) +
  geom_point(data=df2,aes(x=x,y=df2$y2),alpha=2/10, shape=21, fill="green", colour="black", size=5) +
  geom_smooth(data=df1,aes(x=x,y=df1$y1),method = "lm", se = FALSE, formula = my.formula1, colour = "red") +
  geom_smooth(data=df2,aes(x=x,y=df2$y2),method = "lm", se = FALSE, formula = my.formula2, colour = "blue")

m1 <- lm(my.formula1, df1)
m2 <- lm(my.formula2, df2)

my.eq1 <- as.character(signif(as.polynomial(coef(m1)), 3))
my.eq2 <- as.character(signif(as.polynomial(coef(m2)), 3))
label.text1 <- paste(gsub("x", "~italic(x)", my.eq1, fixed = TRUE))
label.text2 <- paste(gsub("x", "~italic(x)", my.eq2, fixed = TRUE))

p + annotate(geom = "text", x = -Inf, y =max(prices_comparison_apartment2_with_outlier)/2 , label = label.text1,
             family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "red")+
  #             family = "serif", hjust = 0, parse = TRUE, size = 4)+
  
  annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment2_with_outlier)/4, label = label.text2, 
           family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "blue")


###############################
###We remove the outliers
###############################

#we create a vector "prices comparison apartment 2 without outliers"
prices_comparison_apartment2_without_outlier <- prices_comparison_apartment2[prices_comparison_apartment2 > -1e+3] 
prices_comparison_apartment2_without_outlier <- prices_comparison_apartment2_without_outlier[!prices_comparison_apartment2_without_outlier > 1e+8] 

#number of outliers removed
nb_outliers_removed_OLS_estimation_apartment_prices_comparison_apartment2=length(prices_comparison_apartment2)-length(prices_comparison_apartment2_without_outlier)

indices_outliers_graph2_apartment_condition_1=which(prices_comparison_apartment2 < -1e+3)
indices_outliers_graph2_apartment_condition_2=which(prices_comparison_apartment2 > 1e+8)

#we create a vector which gathers the elements which shall be removed according to the conditions specified above (outliers)
indices_outliers_graph2_apartment=c(indices_outliers_graph2_apartment_condition_1,indices_outliers_graph2_apartment_condition_2)

data_for_graph2_without_outlier=cbind(1:length(prices_comparison_apartment2_without_outlier),prices_comparison_apartment2_without_outlier)
data_for_graph2_without_outlier=as.data.frame(data_for_graph2_without_outlier)

x <- data.frame(x = 1:length(prices_comparison_apartment2_without_outlier)) 
df1 <- data.frame("x"=x, "y1"=data_for_graph2_without_outlier$prices_comparison_apartment2_without_outlier)

#we need to change the benchmark - we remove the values which correspond to the indices of the outliers
#we recreate a new dataframe adapted
prices_comparison_benchmark_apartment_graph_2_without_outlier=prices_comparison_benchmark_apartment[-indices_outliers_graph2_apartment]
#length(prices_comparison_benchmark_apartment_graph_3)
data_for_graph_2_without_outlier_benchmark=cbind(1:length(prices_comparison_apartment2_without_outlier),prices_comparison_benchmark_apartment_graph_2_without_outlier)
data_for_graph_2_without_outlier_benchmark=as.data.frame(data_for_graph_2_without_outlier_benchmark)

#data_for_benchmark_graph3_apartment=prices_comparison_benchmark_apartment[prices_comparison_benchmark_apartment!=prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
#length(data_for_benchmark_graph3_apartment)
#data_for_benchmark_graph3_apartment=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[-data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
df2 <- data.frame("x"=x, "y2"=data_for_graph_2_without_outlier_benchmark$prices_comparison_benchmark_apartment_graph_2_without_outlier)

#df_benchmark <- data.frame("x"=x, "y"=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment)
my.formula1 <- df1$y1 ~ poly(x, 2, raw = TRUE)
my.formula2 <- df2$y2 ~ poly(x, 2, raw = TRUE)

#my.formula2
#p <- ggplot(df, aes(x, y))
p <- ggplot()+
  #four next lines of the code adapted from http://stackoverflow.com/questions/10438752/adding-x-and-y-axis-labels-in-ggplot2
  scale_size_area() + 
  xlab("Observations") +
  ylab("Prices difference") +
  ggtitle("Prices difference for Property type apartment with adapted Interest Rates - without outliers")+
  geom_point(data=df1,aes(x=x,y=df1$y1),alpha=2/10, shape=21, fill="blue", colour="black", size=5) +
  geom_point(data=df2,aes(x=x,y=df2$y2),alpha=2/10, shape=21, fill="green", colour="black", size=5) +
  geom_smooth(data=df1,aes(x=x,y=df1$y1),method = "lm", se = FALSE, formula = my.formula1, colour = "red") +
  geom_smooth(data=df2,aes(x=x,y=df2$y2),method = "lm", se = FALSE, formula = my.formula2, colour = "blue")

m1 <- lm(my.formula1, df1)
m2 <- lm(my.formula2, df2)

my.eq1 <- as.character(signif(as.polynomial(coef(m1)), 3))
my.eq2 <- as.character(signif(as.polynomial(coef(m2)), 3))
label.text1 <- paste(gsub("x", "~italic(x)", my.eq1, fixed = TRUE))
label.text2 <- paste(gsub("x", "~italic(x)", my.eq2, fixed = TRUE))

p + annotate(geom = "text", x = -Inf, y =max(prices_comparison_apartment2_without_outlier)/2 , label = label.text1,
             family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "red")+
  #             family = "serif", hjust = 0, parse = TRUE, size = 4)+
  
  annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment2_without_outlier)/4, label = label.text2, 
           family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "blue")


############################################################
#II. SIMULATION OF THE CAP RATES WITH A VASICEK MODEL
############################################################

##########################OLS#############################
##################METHOD FOR N PROCESSES##################
##########################################################

#We apply the method given in #Modeling and simulating Interest Rates via Time-Dependent Mean Reversion
#by Andrew Jason Dweck (2014) - pages 15-19

#we have a data frame
data_apartment_matrix_cap_rates
#we convert the data frame into a matrix
data_apartment_matrix_cap_rates_as_matrix=as.matrix(data_apartment_matrix_cap_rates)

#we first define the number of rows and the number of columns
dimension_matrix_cap_rates=dim(data_apartment_matrix_cap_rates)
nb_rows_cap_rates=dimension_matrix_cap_rates[1]
nb_columns_cap_rates=dimension_matrix_cap_rates[2]

#vector of differences - we adapt it to different cap rates - we compute the differences within each vector - it is needed for the calibration
differences_cap_rates=matrix(data=0,nrow=nb_rows_cap_rates, ncol=nb_columns_cap_rates)
for (j in 1:nb_columns_cap_rates) {
  for (i in 2:nb_rows_cap_rates) {
  differences_cap_rates[i,j]=data_apartment_matrix_cap_rates[i,j]-data_apartment_matrix_cap_rates[i-1,j]
}
}

#In order to apply the OLS, we will need to lag the variables. For this purpose, let us convert in this part the series into time series
data_apartment_matrix_cap_rates_as_ts=ts(data_apartment_matrix_cap_rates_as_matrix)

#we also convert the differences cap rates into time series
differences_cap_rates_as_ts=ts(as.matrix(differences_cap_rates))

#we define the fit_multiple_cap_rate as a list
fit_multiple_cap_rates=list()
for (i in 1:nb_columns_cap_rates) {
  fit_multiple_cap_rates[[i]]= 0
}

#we apply the OLS method to time series - we need to lag "data_apartment_matrix_cap_rates_as_ts". The method has been inspired from:
#http://stats.stackexchange.com/questions/92498/forecasting-time-series-regression-in-r-using-lm
for (j in 1:nb_columns_cap_rates) {
  fit_multiple_cap_rates[[j]]<-dynlm(differences_cap_rates_as_ts[,j] ~ L(data_apartment_matrix_cap_rates_as_ts[,j]))
}

#We define the values of the coefficients of the coefficients
Intercept_cap_rates=rep(0, nb_columns_cap_rates)
First_term_cap_rates=rep(0, nb_columns_cap_rates)

for (j in 1:nb_columns_cap_rates) {
  Intercept_cap_rates[j]=fit_multiple_cap_rates[[j]]$coef[[1]]
  First_term_cap_rates[j]=fit_multiple_cap_rates[[j]]$coef[[2]]
}

#we find the estimates of a by dividing the vector b1 by delta t, as expressed in the paper.
#in our case delta t =0.25 as we have for the cap rates an observation every quarter of the year
delta_t=0.25
a_hat_cap_rates_ols= -First_term_cap_rates/delta_t
b_hat_cap_rates_ols=Intercept_cap_rates/(delta_t*a_hat_cap_rates_ols)

#we then compute the one step - prediction equation
#we need to give the initial values of the processes: we always define the initial value of the 
#processes as the cap rate for the first available time value, i.e. 2010 1
fit_multiple_cap_rates_ols=matrix(data=0,nrow=nb_rows_cap_rates, ncol=nb_columns_cap_rates)

for (j in 1:nb_columns_cap_rates) {
  fit_multiple_cap_rates_ols[1,j]=data_apartment_matrix_cap_rates[1,j]
}

for (j in 1:nb_columns_cap_rates) {
  for (i in 2:nb_rows_cap_rates) {
    fit_multiple_cap_rates_ols[i,j]=(1-a_hat_cap_rates_ols[j]*delta_t)*fit_multiple_cap_rates_ols[i-1,j]+a_hat_cap_rates_ols[j]*b_hat_cap_rates_ols[j]*delta_t
  }
}

#we then compute the standard deviation s of the prediction errors
standard_deviation_cap_rates=rep(0,nb_columns_cap_rates)
for (j in 1:nb_columns_cap_rates) {
  standard_deviation_cap_rates[j]=sd(data_apartment_matrix_cap_rates[,j]-fit_multiple_cap_rates_ols[,j])
}

#we can then obtain the value of the estimate of sigma
sigma_hat_cap_rates_ols=rep(0,nb_columns_cap_rates)
sigma_hat_cap_rates_ols=standard_deviation_cap_rates/sqrt(delta_t)

#then we have our equation for predictions:
fit_multiple_cap_rates_tilde=matrix(data=0,nrow=nb_rows_cap_rates, ncol=nb_columns_cap_rates)
#We give the initial value of the process for fit_tilde
for (j in 1:nb_columns_cap_rates) {
  fit_multiple_cap_rates_tilde[1,j]=data_apartment_matrix_cap_rates[1,j]
}

#We compute fit_tilde
for (j in 1:nb_columns_cap_rates) {
  for (i in 2:nb_rows_cap_rates) {
    fit_multiple_cap_rates_tilde[i,j]=(1-a_hat_cap_rates_ols[j]*delta_t)*fit_multiple_cap_rates_tilde[i-1,j]+a_hat_cap_rates_ols[j]*b_hat_cap_rates_ols[j]*delta_t+sigma_hat_cap_rates_ols[j]*sqrt(delta_t)*rnorm(1)
  }
}

#The out of sample predictions are then:
#This vector gives the cap rates we can insert into the "asset paths equation"
fit_multiple_cap_rates_tilde_predictions=rep(0,nb_columns_cap_rates)
for (j in 1:nb_columns_cap_rates) {
  fit_multiple_cap_rates_tilde_predictions[j]=(1-a_hat_cap_rates_ols[j]*delta_t)*fit_multiple_cap_rates_tilde[nb_rows_cap_rates,j]+a_hat_cap_rates_ols[j]*b_hat_cap_rates_ols[j]*delta_t+sigma_hat_cap_rates_ols[j]*sqrt(delta_t)*rnorm(1)
}

#####Several examples of the Vasicek estimation with OLS
#First Property
plot(data_apartment_matrix_cap_rates[,1],main = "", xlab="Observation", ylab="Value of the cap rate")
lines(fit_multiple_cap_rates_tilde[,1],col="blue", lty=2)
title(main="Modelization of the cap rates under a Vasicek process with OLS",sub="Apartment Property Number 1")
legend("topleft", legend=c("Actual values of the cap rates", "Fitted values"), lty=2, cex=0.8, pch=c(1,19),col=c("black","blue"))

#Second Property
plot(data_apartment_matrix_cap_rates[,2],main = "", xlab="Observation", ylab="Value of the cap rate")
lines(fit_multiple_cap_rates_tilde[,2],col="blue", lty=2)
title(main="Modelization of the cap rates under a Vasicek process with OLS",sub="Apartment Property Number 2")
legend("topleft", legend=c("Actual values of the cap rates", "Fitted values"), lty=2, cex=0.8, pch=c(1,19),col=c("black","blue"))

#Compute of the estimates theta1, theta2 and theta3 from the OLS estimates in order to insert them in the formula of the sde package
theta1_Vasicek_OLS=a_hat_cap_rates_ols
theta2_Vasicek_OLS=rep(0,nb_columns_cap_rates)
theta2_Vasicek_OLS=a_hat_cap_rates_ols*b_hat_cap_rates_ols
theta3_Vasicek_OLS=sigma_hat_cap_rates_ols

################MAXIMUM LIKELIHOOD ESTIMATION#############
##################METHOD FOR N PROCESSES##################
##########################################################

#The book is: "Simulation and Inference fir Stochastic Differential Equations
#by Stefano M. Iacus
#Springer Series in Statistics, 2008
#for this purpose we first need to load two functions available page 114
#In the optimization, first guesses are needed. These are taken from the results of the OLS we just obtained. Just for the volatility, we take as initial guesses a volatility of 100
#so that we have convergence.
#The functions are also available in the sde package
#The references for the sde package are the following: https://cran.r-project.org/web/packages/sde/sde.pdf

#first function
dcOu<-function(x,t,x0,theta,log=FALSE) {
  Ex<-theta[1]/theta[2]+(x0-theta[1]/theta[2])*exp(-theta[2]*t)
  Vx<-theta[3]^2*(1-exp(-2*theta[2]*t))/(2*theta[2])
  dnorm(x,mean=Ex,sd=sqrt(Vx),log=log)
}

#second function
OU.lik <- function(theta1,theta2,theta3){
  n <-length(X)
  dt <-deltat(X)
  -sum(dcOU(X[2:n],dt,X[1:(n-1)],c(theta1,theta2,theta3), log=TRUE))
}

#List which is going to contain the results
fit_multiple_Vasicek_MLE=list()
for (i in 1:nb_columns_cap_rates) {
  fit_multiple_Vasicek_MLE[[i]]= 0
}

#We proceed here to the optimization
#sometimes errors happen: the function tryCatch allows the optimization the persevere. Indeed, sometimes the algorithm of optimization
#does not converge.
#for the first guesses we take the estimates obtained from the OLS, otherwise the optimization may not converge
#However, if we take for the initial guesses the value from sigma_OLS, the optimization does not provide solutions
#Consequently, we implement a value of 100 and then get estimates.

#we use the function tryCatch so that the for loop continues looking for estimates, even if there is an error.
for (j in 1:nb_columns_cap_rates){
  X<-as.ts(data_apartment_matrix_cap_rates[,j])
  fit_multiple_Vasicek_MLE[j]<-tryCatch(mle(OU.lik,start=list(theta1=a_hat_cap_rates_ols[j]*b_hat_cap_rates_ols[j],theta2=a_hat_cap_rates_ols[j], theta3=100), method="SANN"))
}

#we give the different parameters estimated
theta1_Vasicek_MLE=rep(0,nb_columns_cap_rates)
theta2_Vasicek_MLE=rep(0,nb_columns_cap_rates)
theta3_Vasicek_MLE=rep(0,nb_columns_cap_rates)

#we give the different coefficients
for (j in 1:nb_columns_cap_rates) {
  theta1_Vasicek_MLE[j]=fit_multiple_Vasicek_MLE[[j]]@fullcoef[[1]]
  theta2_Vasicek_MLE[j]=fit_multiple_Vasicek_MLE[[j]]@fullcoef[[2]]  
  theta3_Vasicek_MLE[j]=fit_multiple_Vasicek_MLE[[j]]@fullcoef[[3]]
}


###Then we need to simulate the processes with these parameters as the basis

##We simulate the different Vasicek processes for the cap rates with the MLE estimators
#Initialisation of the variable
Simulation_cap_rates_OLS=list()
Simulation_cap_rates_MLE=list()

#Simulation of the process
for (j in 1:nb_columns_cap_rates) {
  Simulation_cap_rates_OLS[[j]]=sde.sim(model="OU", theta=c(theta1_Vasicek_OLS[j],theta2_Vasicek_OLS[j],theta3_Vasicek_OLS[j]),N=10,T=apartment_time_difference[j])
  Simulation_cap_rates_MLE[[j]]=sde.sim(model="OU", theta=c(theta1_Vasicek_MLE[j],theta2_Vasicek_MLE[j],theta3_Vasicek_MLE[j]),N=10,T=apartment_time_difference[j])
}

#we take the last coefficient of the list "Simulation cap rates"
#as we did 10 simulations for each cap rate, we shall take the last simulation (the 10th)
Values_cap_rates_Vasicek_model_to_be_inserted_formula_asset_paths_OLS=rep(0,nb_columns_cap_rates)
Values_cap_rates_Vasicek_model_to_be_inserted_formula_asset_paths_MLE=rep(0,nb_columns_cap_rates)

#We have the different values for the simulated cap rates depending on the type of estimation:

######AS WE WORKED WITH PERCENTAGES, WE ALWAYS DIVIDE BY 100!!!!!!!
for (j in 1:nb_columns_cap_rates) {
  Values_cap_rates_Vasicek_model_to_be_inserted_formula_asset_paths_OLS[[j]]=Simulation_cap_rates_OLS[[j]][10]/100
  Values_cap_rates_Vasicek_model_to_be_inserted_formula_asset_paths_MLE[[j]]=Simulation_cap_rates_MLE[[j]][10]/100
}

#####Several examples of the Vasicek estimation with MLE

#we quickly redo the simuations for the graph and change the number of iterations and adapt them to the length of the cap rates
Simulation_cap_rates_MLE_for_graphs=list()

for (j in 1:2) {
  Simulation_cap_rates_MLE_for_graphs[[j]]=sde.sim(model="OU", theta=c(theta1_Vasicek_MLE[j],theta2_Vasicek_MLE[j],theta3_Vasicek_MLE[j]),N=length(data_apartment_matrix_cap_rates[,1]),T=apartment_time_difference[j])
}

#First Property
plot(data_apartment_matrix_cap_rates[,1],main = "", xlab="Observation", ylab="Value of the cap rate",ylim=c(-0.03, 0.2))
lines(as.vector(Simulation_cap_rates_MLE_for_graphs[[1]]/100),col="blue", lty=2)
title(main="Modelization of the cap rates under a Vasicek process with MLE",sub="Apartment Property Number 1")
legend("topleft", legend=c("Actual values of the cap rates", "Fitted values"), lty=2, cex=0.8, pch=c(1,19),col=c("black","blue"))

#Second Property
plot(data_apartment_matrix_cap_rates[,2],main = "", xlab="Observation", ylab="Value of the cap rate",ylim=c(-0.03, 0.2))
lines(as.vector(Simulation_cap_rates_MLE_for_graphs[[2]]/100),col="blue", lty=2)
title(main="Modelization of the cap rates under a Vasicek process with MLE",sub="Apartment Property Number 2")
legend("topleft", legend=c("Actual values of the cap rates", "Fitted values"), lty=2, cex=0.8, pch=c(1,19),col=c("black","blue"))

############################################################
#III. SIMULATION OF THE INTEREST RATES UNDER A CIR MODEL
############################################################

#We load the data of the three months maturity Treasury Rates
interest_rates_3_months=read.xlsx("Risk_free_rate_FED.xlsx", sheet = "Data", startRow = 1, colNames = TRUE,
                                  rowNames =FALSE, detectDates = FALSE, skipEmptyRows = TRUE,
                                  rows = NULL, cols = 1:2, check.names = FALSE, namedRegion = NULL)

#When loaded, the expression are expressed as percentages.
#We will in the end divide by 100 the final values found in order to insert the values into the Geometric Brownian Motion which simulates asset prices, as, according to
#Hull (Options, Futures and Other Derivatives, 8th edition, Prentice Hall, page 287), the input parameters need to be expressed as numbers and not percentages
#Besides, when doind the MLE estimation, Stefano M. Iacus uses values expressed in percentages as it can be seen page 20 in:
#http://www2.ms.u-tokyo.ac.jp/probstat/?action=multidatabase_action_main_filedownload&download_flag=1&upload_id=122&metadata_id=277
#The complete references of this document are: "The likelihood function" by S.M Iacus 2014

############################################################################
###############################OLS CASE#####################################
############################################################################

#Similarly to the Vasicek case and the cap rates, we apply for getting the OLS estimates the methodology given in:
#Modeling and simulating Interest Rates via Time-Dependent Mean Reversion
#by Andrew Jason Dweck (2014) - pages 15-19
#However, in this case we do it for the CIR process
dimension_matrix_interest_rates_3_months=dim(interest_rates_3_months)
nb_rows_interest_rates=dimension_matrix_interest_rates_3_months[1]
nb_columns_interest_rates=dimension_matrix_interest_rates_3_months[2]

#We load the variable r_i
r_i=rep(0,nb_rows_cap_rates)
for (i in 1:nb_rows_cap_rates) {
  r_i[i]=interest_rates_3_months[i,2]
}

#we create the vector of weighted differences
weighted_differences_r_i=rep(0,length(r_i)-1)
for (i in 2:length(r_i)) {
  weighted_differences_r_i[i-1]=(r_i[i]-r_i[i-1])/sqrt(r_i[i-1])
}

#we convert r_i and the weighted differences into time series, in order to do the OLS regression with lagged variables
r_i_ts=ts(r_i)
weighted_differences_r_i_ts=ts(weighted_differences_r_i)

#we apply the OLS method
fit <- dynlm(weighted_differences_r_i_ts~1/sqrt(L(r_i_ts))+sqrt(L(r_i_ts))) 

#then we get the coeficients:
#coefficient b0
b1=fit$coefficients[[1]]

#coefficient b1
b2=fit$coefficients[[2]]

#we define delta t. In our case:
delta_t=0.25
#estimate for a - we divide it by minus delta t
a_hat_interest_rates_OLS=-b2/delta_t

#estimate for b - we divide b0 by a_hat and delta t
b_hat_interest_rates_OLS=b1/(a_hat_interest_rates_OLS*delta_t)

#we now have r_i_hat:
r_i_hat=rep(0,length(r_i))
r_i_hat[1]=r_i[1]
for (i in 2:length(r_i)) {
  r_i_hat[i]=(1-a_hat_interest_rates_OLS*delta_t)*r_i[i-1]+a_hat_interest_rates_OLS*b_hat_interest_rates_OLS*delta_t
}

#we then compute the standard deviation of the prediction erros
#we first need to define r_i_hat as a time series
r_i_hat_ts=ts(r_i_hat)
#we then compute the standard deviation - we have lagged r_i_ts as suggested in the paper
s_CIR_ts=sd((r_i_hat_ts-r_i_ts)/lag(r_i_ts))
  
#we then obtain the volatility
#we need first to convert the time series s_CIR to a vector
s_CIR=as.numeric(s_CIR_ts)
sigma_hat_interest_rates_OLS=s_CIR/sqrt(delta_t)

#so we can now simulate the whole process
r_i_tilde=rep(0,length(r_i))
#initial value of the process
r_i_tilde[1]=r_i[1]
#Simulation of the process
for (i in 2:length(r_i)) {
  r_i_tilde[i]=(1-a_hat_interest_rates_OLS*delta_t)*r_i_tilde[i-1]+a_hat_interest_rates_OLS*b_hat_interest_rates_OLS*delta_t+sigma_hat_interest_rates_OLS*sqrt(delta_t)*rnorm(1)
}

#We convert the coefficients estimated into theta1, theta2 and theta3: that will allow us to simulate the CIR processes according to the sde package
theta1_CIR_OLS=a_hat_interest_rates_OLS
theta2_CIR_OLS=a_hat_interest_rates_OLS*b_hat_interest_rates_OLS
theta3_CIR_OLS=sigma_hat_interest_rates_OLS
##We shall not make predictions with this method as the period are not adpated

#the interest of this method relies on the prediction of the next value. We have:
r_i_tilde_prediction_OLS=(1-a_hat_interest_rates_OLS*delta_t)*r_i_tilde[length(r_i)]+a_hat_interest_rates_OLS*b_hat_interest_rates_OLS*delta_t+sigma_hat_interest_rates_OLS*sqrt(delta_t)*rnorm(1)

#As the values of the interest rates loaded were expressed as percentages, we need to divide the value by 100 in order to be consistent.
r_i_tilde_prediction_OLS=r_i_tilde_prediction_OLS/100

############################################################################
##########MAXIMUM LIKELIHOOD ESTIMATION#####################################
############We load functions from the sde package##########################

#Procedure adapted from Stefano M.Iacus, "Simulation and Inference for Stochastic Differential Equations", Springer Series in Statistics, 2008
#Procedure adapted from "The likelihood function" by S.M Iacus 2014
#The document is available at the following URL: http://www2.ms.u-tokyo.ac.jp/probstat/?action=multidatabase_action_main_filedownload&download_flag=1&upload_id=122&metadata_id=277
#The yuima package is used and the quasi maximum likelihood estimation is also applied.
#The following book has also been used:
#Stefano M.Iacus, "Simulation and Inference for Stochastic Differential Equations, Springer Series in Statistics, 2008
#The following document has also been used for simulating the processes:
#https://www.rmetrics.org/files/Meielisalp2009/Presentations/Iacus.pdf
#The complete references are: "From the sde package to the Yuima Project", by S.M. Iacus, R/RMetrics Workshop, June 28th - July 2nd, 2009

#3 months maturity Treasury bill
X_interest_rates_3_months=interest_rates_3_months[,2]
time_interest_rate=interest_rates_3_months[,1]

###########################################################################################################
#We load several functions - pages 119 121 "Simulation and Inference for Stochastic Differential Equations"
###########################################################################################################

#The exponentially rescalled Bessel function
expBes <- function (x,nu ){#
  mu <- 4*nu ^2
  A1 <- 1
  A2 <- A1 * (mu - 1) / (1 * (8*x))
  A3 <- A2 * (mu - 9) / (2 * (8*x))
  A4 <- A3 * (mu - 25) / (3 * (8*x))
  A5 <- A4 * (mu - 49) / (4 * (8*x))
  A6 <- A5 * (mu - 81) / (5 * (8*x))
  A7 <- A6 * (mu -121) / (6 * (8*x))
  1/ sqrt (2*pi*x) * (A1 - A2 + A3 - A4 + A5 - A6 + A7)
}

dcCIR <- function (x, t, x0 , theta , log = FALSE ){
  c <- 2* theta [2] /((1 - exp (- theta [2] *t))* theta [3]^2)
  ncp <- 2*c*x0* exp (- theta [2] *t)
  df <- 4* theta [1] / theta [3]^2
  u <- c*x0* exp (- theta [2] *t)
  v <- c*x
  q <- 2* theta [1] / theta [3]^2 -1
  lik <- ( log (c) - (u+v) + q/2 * log (v/u) + log ( expBes ( 2* sqrt (u*v), q))
           + 2* sqrt (u*v))
  if(! log )
    lik <- exp ( lik )
  lik
}

CIR.lik <- function ( theta1 , theta2 , theta3 ) {
  n <- length (X)
  dt <- deltat (X)
  -sum ( dcCIR (x=X [2: n], t=dt , x0=X [1:(n -1)] , theta =c( theta1 , theta2 , theta3 ),
                log = TRUE ))
}

# inefficient version based on noncentral chi squared distribution
dcCIR2 <- function (x, t, x0 , theta , log = FALSE )
{
  c <- 2* theta [2] /((1 - exp (- theta [2] *t))* theta [3]^2)
  ncp <- 2*c*x0* exp (- theta [2] *t)
  df <- 4* theta [1] / theta [3]^2
  lik <- ( dchisq (2 * x * c, df = df , ncp = ncp , log = TRUE )
           + log (2*c))
  if(! log )
    lik <- exp ( lik )
  lik
}
CIR.lik2 <- function ( theta1 , theta2 , theta3 ) {
  n <- length (X)
  dt <- deltat (X)
  -sum ( dcCIR2 (x=X [2: n], t=dt , x0=X [1:(n -1)] , theta =c( theta1 , theta2 , theta3 ),
                 log = TRUE ))
}

X=X_interest_rates_3_months
#One of the two estimation may fail as sometimes the approximation tends to explode
fit_CIR_MLE=mle(CIR.lik, start =list(theta1 =.1 , theta2 =.1 , theta3 =.3) , method ="L-BFGS-B", lower =c (0.001 ,0.001 ,0.001) , upper =c(1 ,1 ,1)) 
#In the case of apartments, the first estimation fails but the chi-square approximation succeeds. In the book from Iacus, the following happened.
#fit_CIR_MLE_2=mle(CIR.lik2 , start = list (theta1 =.1 , theta2 =.1 , theta3 =.3) ,method ="L-BFGS-B",lower =c (0.001 ,0.001 ,0.001) , upper =c(1 ,1 ,1)) 
fit_CIR_MLE_2=mle(CIR.lik2 , start = list (theta1 =theta1_CIR_OLS, theta2 =theta2_CIR_OLS , theta3 =theta3_CIR_OLS) ,method ="L-BFGS-B",lower =c (0.001 ,0.001 ,0.001) , upper =c(1 ,1 ,1)) 

#The coefficients are: 
theta1_CIR_MLE_2=coef(fit_CIR_MLE_2)[[1]]
theta2_CIR_MLE_2=coef(fit_CIR_MLE_2)[[2]]
theta3_CIR_MLE_2=coef(fit_CIR_MLE_2)[[3]]

#######################################
###SIMULATION OF THE CIR PROCESSES
#######################################

Simulation_CIR_OLS=list()
Simulation_CIR_MLE_2=list()
for (j in 1:nb_columns_cap_rates) {
  Simulation_CIR_OLS[[j]]=sde.sim(model="CIR", theta=c(theta1_CIR_OLS,theta2_CIR_OLS,theta3_CIR_OLS),N=10,T=apartment_time_difference[j])
  Simulation_CIR_MLE_2[[j]]=sde.sim(model="CIR", theta=c(theta1_CIR_MLE_2,theta2_CIR_MLE_2,theta3_CIR_MLE_2),N=10,T=apartment_time_difference[j])
}

##We create graphs
#For the graph, we adapt the number of observations to the length of the interest rates
Simulation_CIR_OLS_test=list()
Simulation_CIR_MLE_2_test=list()
Simulation_CIR_OLS_test[[1]]=sde.sim(model="CIR", theta=c(theta1_CIR_OLS,theta2_CIR_OLS,theta3_CIR_OLS),N=length(X_interest_rates_3_months),T=apartment_time_difference[length(apartment_time_difference)])
Simulation_CIR_MLE_2_test[[1]]=sde.sim(model="CIR", theta=c(theta1_CIR_MLE_2,theta2_CIR_MLE_2,theta3_CIR_MLE_2),N=length(X_interest_rates_3_months),T=apartment_time_difference[length(apartment_time_difference)])

Values_for_graph_CIR_estimation_OLS=rep(0,length(X_interest_rates_3_months))
Values_for_graph_CIR_estimation_MLE_2=rep(0,length(X_interest_rates_3_months))

for (j in 1:length(X_interest_rates_3_months)) {
  Values_for_graph_CIR_estimation_OLS[[j]]=Simulation_CIR_OLS_test[[1]][j]
  Values_for_graph_CIR_estimation_MLE_2[[j]]=Simulation_CIR_MLE_2_test[[1]][j]
}

#Plot estimation OLS
plot(X_interest_rates_3_months,main = "", xlab="Observation", ylab="Values of the interest rates")
lines(Values_for_graph_CIR_estimation_OLS,col="blue", lty=2)
title(main="Modelization of the interest rates under a CIR process",sub="OLS estimation")
legend("topleft", legend=c("Actual values of the interest rates", "Fitted values"), lty=2, cex=0.8, pch=c(1,19),col=c("black","blue"))

#Plot estimation MLE
plot(X_interest_rates_3_months,main = "", xlab="Observation", ylab="Values of the interest rates")
lines(Values_for_graph_CIR_estimation_MLE_2,col="blue", lty=2)
title(main="Modelization of the interest rates under a CIR process",sub="MLE estimation")
legend("topleft", legend=c("Actual values of the interest rates", "Fitted values"), lty=2, cex=0.8, pch=c(1,19),col=c("black","blue"))

#We have the different values for the simulated processes
#We divide the values by 100 in order to insert them into the Geometric Brownian Motion for the asset prices
Values_interest_rates_CIR_model_to_be_inserted_formula_asset_paths_OLS=rep(0,nb_columns_cap_rates)
Values_interest_rates_CIR_model_to_be_inserted_formula_asset_paths_MLE=rep(0,nb_columns_cap_rates)
for (j in 1:nb_columns_cap_rates) {
  Values_interest_rates_CIR_model_to_be_inserted_formula_asset_paths_OLS[[j]]=Simulation_CIR_OLS[[j]][10]/100
  Values_interest_rates_CIR_model_to_be_inserted_formula_asset_paths_MLE[[j]]=Simulation_CIR_MLE_2[[j]][10]/100
}

############
###Compute of the new returns, based on the Vasicek model for the cap rates and CIR model for the Interest rates, as suggested in 
#Goldbeck and Linetsky, Least Squares Monte Carlo Valuation of Residential Mortgages with Prepayment and Default Risk, available at the following URL:
#http://faculty.washington.edu/golbeck/MortgageSim.pdf

apartment_returns_cap_rates_OLS_interest_rates_OLS=Values_interest_rates_CIR_model_to_be_inserted_formula_asset_paths_OLS-Values_cap_rates_Vasicek_model_to_be_inserted_formula_asset_paths_OLS
apartment_returns_cap_rates_MLE_interest_rates_MLE=Values_interest_rates_CIR_model_to_be_inserted_formula_asset_paths_MLE-Values_cap_rates_Vasicek_model_to_be_inserted_formula_asset_paths_MLE

###We can now simulate the new asset paths based on the Vasicek model for the cap rates and the CIR process for the interest rates
prices_apartment3=asset.paths(apartment_Init_acq_cost, apartment_returns_cap_rates_OLS_interest_rates_OLS, covariance_matrix_apartment, 100, periods=apartment_time_difference)
prices_apartment4=asset.paths(apartment_Init_acq_cost, apartment_returns_cap_rates_MLE_interest_rates_MLE, covariance_matrix_apartment, 100, periods=apartment_time_difference)

##We take the average price in the end
average_asset_price_apartment3=rep(0,length(apartment_dividends))
average_asset_price_apartment4=rep(0,length(apartment_dividends))
for (i in 1:length(apartment_dividends)){
  average_asset_price_apartment3[i]=mean(prices_apartment3[i,length(apartment_time_difference),])
  average_asset_price_apartment4[i]=mean(prices_apartment4[i,length(apartment_time_difference),])
}

#we create a vector in which there is a price comparison between the actual prices and the prices simulated
prices_comparison_apartment3=apartment_NetSalePrice-average_asset_price_apartment3
prices_comparison_apartment4=apartment_NetSalePrice-average_asset_price_apartment4

#then we draw graphs to compare the actual prices and the prices obtained after Monte-Carlo simulations
#the present application of the package ggplot2 has been taken from:
#http://stackoverflow.com/questions/11949331/adding-a-3rd-order-polynomial-and-its-equation-to-a-ggplot-in-r
data_for_graph3=cbind(1:length(prices_comparison_apartment3),prices_comparison_apartment3)
data_for_graph4=cbind(1:length(prices_comparison_apartment4),prices_comparison_apartment4)

#we convert the data into a data frame - this is needed for a good application of the graphs
data_for_graph3=as.data.frame(data_for_graph3)
data_for_graph4=as.data.frame(data_for_graph4)

#########################
#Graphs
#########################

#has been adapted from:
#http://stackoverflow.com/questions/11949331/adding-a-3rd-order-polynomial-and-its-equation-to-a-ggplot-in-r

#########
#Graph 3 - Prices Comparison when the cap rates are simulated with a Vasicek model and the Interest rates with a CIR model - OLS estimation type
#########

###############################
###We keep the outliers
###############################

#case we want the graph with outliers: we put extreme limits
prices_comparison_apartment3_with_outlier <- prices_comparison_apartment3[prices_comparison_apartment3 > -1e+200] 
prices_comparison_apartment3_with_outlier <- prices_comparison_apartment3_with_outlier[!prices_comparison_apartment3_with_outlier > 1e+200] 

#conditions to produce the graph with outliers
indices_outliers_graph3_apartment_condition_1=which(prices_comparison_apartment3 < -1e+200)
indices_outliers_graph3_apartment_condition_2=which(prices_comparison_apartment3 > 1e+200)
#we create a vector which gathers the elements which shall be removed according to the conditions specified above (outliers)
indices_outliers_graph3_apartment=c(indices_outliers_graph3_apartment_condition_1,indices_outliers_graph3_apartment_condition_2)

data_for_graph3_with_outlier=cbind(1:length(prices_comparison_apartment3_with_outlier),prices_comparison_apartment3_with_outlier)
data_for_graph3_with_outlier=as.data.frame(data_for_graph3_with_outlier)

x <- data.frame(x = 1:length(prices_comparison_apartment3_with_outlier)) 
df1 <- data.frame("x"=x, "y1"=data_for_graph3_with_outlier$prices_comparison_apartment3_with_outlier)

#we need to change the benchmark - we remove the values which correspond to the indices of the outliers
#we recreate a new dataframe adapted
#prices_comparison_benchmark_apartment
prices_comparison_benchmark_apartment_graph_3_with_outlier=prices_comparison_benchmark_apartment
#length(prices_comparison_benchmark_apartment_graph_3)
data_for_graph_3_with_outlier_benchmark=cbind(1:length(prices_comparison_apartment3_with_outlier),prices_comparison_benchmark_apartment_graph_3_with_outlier)
data_for_graph_3_with_outlier_benchmark=as.data.frame(data_for_graph_3_with_outlier_benchmark)

#data_for_benchmark_graph3_apartment=prices_comparison_benchmark_apartment[prices_comparison_benchmark_apartment!=prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
#length(data_for_benchmark_graph3_apartment)
#data_for_benchmark_graph3_apartment=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[-data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
df2 <- data.frame("x"=x, "y2"=data_for_graph_3_with_outlier_benchmark$prices_comparison_benchmark_apartment_graph_3_with_outlier)

#df_benchmark <- data.frame("x"=x, "y"=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment)
my.formula1 <- df1$y1 ~ poly(x, 2, raw = TRUE)
my.formula2 <- df2$y2 ~ poly(x, 2, raw = TRUE)

#my.formula2
#p <- ggplot(df, aes(x, y))
p <- ggplot()+
  #four next lines of the code adapted from http://stackoverflow.com/questions/10438752/adding-x-and-y-axis-labels-in-ggplot2
  scale_size_area() + 
  xlab("Observations") +
  ylab("Prices difference") +
  ggtitle("Prices differences: cap rates Vasicek model Interest rates CIR model - OLS estimation type - with outliers")+
  geom_point(data=df1,aes(x=x,y=df1$y1),alpha=2/10, shape=21, fill="blue", colour="black", size=5) +
  geom_point(data=df2,aes(x=x,y=df2$y2),alpha=2/10, shape=21, fill="green", colour="black", size=5) +
  geom_smooth(data=df1,aes(x=x,y=df1$y1),method = "lm", se = FALSE, formula = my.formula1, colour = "red") +
  geom_smooth(data=df2,aes(x=x,y=df2$y2),method = "lm", se = FALSE, formula = my.formula2, colour = "blue")

m1 <- lm(my.formula1, df1)
m2 <- lm(my.formula2, df2)

my.eq1 <- as.character(signif(as.polynomial(coef(m1)), 3))
my.eq2 <- as.character(signif(as.polynomial(coef(m2)), 3))
label.text1 <- paste(gsub("x", "~italic(x)", my.eq1, fixed = TRUE))
label.text2 <- paste(gsub("x", "~italic(x)", my.eq2, fixed = TRUE))

p + annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment3_with_outlier)/2 , label = label.text1,
             family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "red")+
  #             family = "serif", hjust = 0, parse = TRUE, size = 4)+
  
  annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment3_with_outlier)/4, label = label.text2, 
           family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "blue")


###############################
###We remove the outliers
###############################

#we create a vector "prices comparison apartment 3 without outliers"
prices_comparison_apartment3_without_outlier <- prices_comparison_apartment3[prices_comparison_apartment3 > -1e+3] 
prices_comparison_apartment3_without_outlier <- prices_comparison_apartment3_without_outlier[!prices_comparison_apartment3_without_outlier > 1e+8] 

#number of outliers removed
nb_outliers_removed_OLS_estimation_apartment_prices_comparison_apartment3=length(prices_comparison_apartment3)-length(prices_comparison_apartment3_without_outlier)

indices_outliers_graph3_apartment_condition_1=which(prices_comparison_apartment3 < -1e+3)
indices_outliers_graph3_apartment_condition_2=which(prices_comparison_apartment3 > 1e+8)
#conditions to produce the graph with outliers
#indices_outliers_graph3_apartment_condition_1=which(prices_comparison_apartment3 < -1e+200)
#indices_outliers_graph3_apartment_condition_2=which(prices_comparison_apartment3 > 1e+200)
#we create a vector which gathers the elements which shall be removed according to the conditions specified above (outliers)
indices_outliers_graph3_apartment=c(indices_outliers_graph3_apartment_condition_1,indices_outliers_graph3_apartment_condition_2)

data_for_graph3_without_outlier=cbind(1:length(prices_comparison_apartment3_without_outlier),prices_comparison_apartment3_without_outlier)
data_for_graph3_without_outlier=as.data.frame(data_for_graph3_without_outlier)

x <- data.frame(x = 1:length(prices_comparison_apartment3_without_outlier)) 
df1 <- data.frame("x"=x, "y1"=data_for_graph3_without_outlier$prices_comparison_apartment3_without_outlier)

#we need to change the benchmark - we remove the values which correspond to the indices of the outliers
#we recreate a new dataframe adapted
#prices_comparison_benchmark_apartment
prices_comparison_benchmark_apartment_graph_3_without_outlier=prices_comparison_benchmark_apartment[-indices_outliers_graph3_apartment]
#length(prices_comparison_benchmark_apartment_graph_3)
data_for_graph_3_without_outlier_benchmark=cbind(1:length(prices_comparison_apartment3_without_outlier),prices_comparison_benchmark_apartment_graph_3_without_outlier)
data_for_graph_3_without_outlier_benchmark=as.data.frame(data_for_graph_3_without_outlier_benchmark)

#data_for_benchmark_graph3_apartment=prices_comparison_benchmark_apartment[prices_comparison_benchmark_apartment!=prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
#length(data_for_benchmark_graph3_apartment)
#data_for_benchmark_graph3_apartment=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[-data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[indices_outliers_graph3_apartment]]
df2 <- data.frame("x"=x, "y2"=data_for_graph_3_without_outlier_benchmark$prices_comparison_benchmark_apartment_graph_3_without_outlier)

#df_benchmark <- data.frame("x"=x, "y"=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment)
my.formula1 <- df1$y1 ~ poly(x, 2, raw = TRUE)
my.formula2 <- df2$y2 ~ poly(x, 2, raw = TRUE)

#my.formula2
#p <- ggplot(df, aes(x, y))
p <- ggplot()+
  #four next lines of the code adapted from http://stackoverflow.com/questions/10438752/adding-x-and-y-axis-labels-in-ggplot2
  scale_size_area() + 
  xlab("Observations") +
  ylab("Prices difference") +
  ggtitle("Prices differences: cap rates Vasicek model Interest rates CIR model - OLS estimation type - without outliers")+
  geom_point(data=df1,aes(x=x,y=df1$y1),alpha=2/10, shape=21, fill="blue", colour="black", size=5) +
  geom_point(data=df2,aes(x=x,y=df2$y2),alpha=2/10, shape=21, fill="green", colour="black", size=5) +
  geom_smooth(data=df1,aes(x=x,y=df1$y1),method = "lm", se = FALSE, formula = my.formula1, colour = "red") +
  geom_smooth(data=df2,aes(x=x,y=df2$y2),method = "lm", se = FALSE, formula = my.formula2, colour = "blue")

m1 <- lm(my.formula1, df1)
m2 <- lm(my.formula2, df2)

my.eq1 <- as.character(signif(as.polynomial(coef(m1)), 3))
my.eq2 <- as.character(signif(as.polynomial(coef(m2)), 3))
label.text1 <- paste(gsub("x", "~italic(x)", my.eq1, fixed = TRUE))
label.text2 <- paste(gsub("x", "~italic(x)", my.eq2, fixed = TRUE))

p + annotate(geom = "text", x = -Inf, y =max(prices_comparison_apartment3_without_outlier)/2 , label = label.text1,
             family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "red")+
  #             family = "serif", hjust = 0, parse = TRUE, size = 4)+
  
  annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment3_without_outlier)/4, label = label.text2, 
           family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "blue")


#########
#Graph 4 - Prices Comparison when the cap rates are simulated with a Vasicek model and the Interest rates with a CIR model - MLE estimation type
########

###############################
###We keep the outliers
###############################

#case we want the graph with outliers: we put extreme limits
prices_comparison_apartment4_with_outlier <- prices_comparison_apartment4[prices_comparison_apartment4 > -1e+200] 
prices_comparison_apartment4_with_outlier <- prices_comparison_apartment4_with_outlier[!prices_comparison_apartment4_with_outlier > 1e+200] 

#conditions to produce the graph with outliers
indices_outliers_graph4_apartment_condition_1=which(prices_comparison_apartment4 < -1e+200)
indices_outliers_graph4_apartment_condition_2=which(prices_comparison_apartment4 > 1e+200)
#we create a vector which gathers the elements which shall be removed according to the conditions specified above (outliers)
indices_outliers_graph4_apartment=c(indices_outliers_graph4_apartment_condition_1,indices_outliers_graph4_apartment_condition_2)

data_for_graph4_with_outlier=cbind(1:length(prices_comparison_apartment4_with_outlier),prices_comparison_apartment4_with_outlier)
data_for_graph4_with_outlier=as.data.frame(data_for_graph4_with_outlier)

x <- data.frame(x = 1:length(prices_comparison_apartment4_with_outlier)) 
df1 <- data.frame("x"=x, "y1"=data_for_graph4_with_outlier$prices_comparison_apartment4_with_outlier)

#we need to change the benchmark - we remove the values which correspond to the indices of the outliers
#we recreate a new dataframe adapted
#prices_comparison_benchmark_apartment
prices_comparison_benchmark_apartment_graph_4_with_outlier=prices_comparison_benchmark_apartment
#length(prices_comparison_benchmark_apartment_graph_4)
data_for_graph_4_with_outlier_benchmark=cbind(1:length(prices_comparison_apartment4_with_outlier),prices_comparison_benchmark_apartment_graph_4_with_outlier)
data_for_graph_4_with_outlier_benchmark=as.data.frame(data_for_graph_4_with_outlier_benchmark)

#data_for_benchmark_graph4_apartment=prices_comparison_benchmark_apartment[prices_comparison_benchmark_apartment!=prices_comparison_benchmark_apartment[indices_outliers_graph4_apartment]]
#length(data_for_benchmark_graph4_apartment)
#data_for_benchmark_graph4_apartment=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[-data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[indices_outliers_graph4_apartment]]
df2 <- data.frame("x"=x, "y2"=data_for_graph_4_with_outlier_benchmark$prices_comparison_benchmark_apartment_graph_4_with_outlier)

#df_benchmark <- data.frame("x"=x, "y"=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment)
my.formula1 <- df1$y1 ~ poly(x, 2, raw = TRUE)
my.formula2 <- df2$y2 ~ poly(x, 2, raw = TRUE)

#my.formula2
#p <- ggplot(df, aes(x, y))
p <- ggplot()+
  #four next lines of the code adapted from http://stackoverflow.com/questions/10448752/adding-x-and-y-axis-labels-in-ggplot2
  scale_size_area() + 
  xlab("Observations") +
  ylab("Prices difference") +
  ggtitle("Prices differences: cap rates Vasicek model Interest rates CIR model - MLE estimation type - with outliers")+
  geom_point(data=df1,aes(x=x,y=df1$y1),alpha=2/10, shape=21, fill="blue", colour="black", size=5) +
  geom_point(data=df2,aes(x=x,y=df2$y2),alpha=2/10, shape=21, fill="green", colour="black", size=5) +
  geom_smooth(data=df1,aes(x=x,y=df1$y1),method = "lm", se = FALSE, formula = my.formula1, colour = "red") +
  geom_smooth(data=df2,aes(x=x,y=df2$y2),method = "lm", se = FALSE, formula = my.formula2, colour = "blue")

m1 <- lm(my.formula1, df1)
m2 <- lm(my.formula2, df2)

my.eq1 <- as.character(signif(as.polynomial(coef(m1)), 4))
my.eq2 <- as.character(signif(as.polynomial(coef(m2)), 4))
label.text1 <- paste(gsub("x", "~italic(x)", my.eq1, fixed = TRUE))
label.text2 <- paste(gsub("x", "~italic(x)", my.eq2, fixed = TRUE))

p + annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment4_with_outlier)/2 , label = label.text1,
             family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "red")+
  #             family = "serif", hjust = 0, parse = TRUE, size = 4)+
  
  annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment4_with_outlier)/4, label = label.text2, 
           family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "blue")


###############################
###We remove the outliers
###############################

#we create a vector "prices comparison apartment 4 without outliers"
prices_comparison_apartment4_without_outlier <- prices_comparison_apartment4[prices_comparison_apartment4 > -1e+3] 
prices_comparison_apartment4_without_outlier <- prices_comparison_apartment4_without_outlier[!prices_comparison_apartment4_without_outlier > 1e+8] 

#number of outliers removed
nb_outliers_removed_OLS_estimation_apartment_prices_comparison_apartment4=length(prices_comparison_apartment4)-length(prices_comparison_apartment4_without_outlier)

indices_outliers_graph4_apartment_condition_1=which(prices_comparison_apartment4 < -1e+3)
indices_outliers_graph4_apartment_condition_2=which(prices_comparison_apartment4 > 1e+8)
#conditions to produce the graph with outliers
#indices_outliers_graph4_apartment_condition_1=which(prices_comparison_apartment4 < -1e+200)
#indices_outliers_graph4_apartment_condition_2=which(prices_comparison_apartment4 > 1e+200)
#we create a vector which gathers the elements which shall be removed according to the conditions specified above (outliers)
indices_outliers_graph4_apartment=c(indices_outliers_graph4_apartment_condition_1,indices_outliers_graph4_apartment_condition_2)

data_for_graph4_without_outlier=cbind(1:length(prices_comparison_apartment4_without_outlier),prices_comparison_apartment4_without_outlier)
data_for_graph4_without_outlier=as.data.frame(data_for_graph4_without_outlier)

x <- data.frame(x = 1:length(prices_comparison_apartment4_without_outlier)) 
df1 <- data.frame("x"=x, "y1"=data_for_graph4_without_outlier$prices_comparison_apartment4_without_outlier)

#we need to change the benchmark - we remove the values which correspond to the indices of the outliers
#we recreate a new dataframe adapted
#prices_comparison_benchmark_apartment
prices_comparison_benchmark_apartment_graph_4_without_outlier=prices_comparison_benchmark_apartment[-indices_outliers_graph4_apartment]
#length(prices_comparison_benchmark_apartment_graph_4)
data_for_graph_4_without_outlier_benchmark=cbind(1:length(prices_comparison_apartment4_without_outlier),prices_comparison_benchmark_apartment_graph_4_without_outlier)
data_for_graph_4_without_outlier_benchmark=as.data.frame(data_for_graph_4_without_outlier_benchmark)

#data_for_benchmark_graph4_apartment=prices_comparison_benchmark_apartment[prices_comparison_benchmark_apartment!=prices_comparison_benchmark_apartment[indices_outliers_graph4_apartment]]
#length(data_for_benchmark_graph4_apartment)
#data_for_benchmark_graph4_apartment=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[-data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment[indices_outliers_graph4_apartment]]
df2 <- data.frame("x"=x, "y2"=data_for_graph_4_without_outlier_benchmark$prices_comparison_benchmark_apartment_graph_4_without_outlier)

#df_benchmark <- data.frame("x"=x, "y"=data_for_graph_benchmark_apartment$prices_comparison_benchmark_apartment)
my.formula1 <- df1$y1 ~ poly(x, 2, raw = TRUE)
my.formula2 <- df2$y2 ~ poly(x, 2, raw = TRUE)

#my.formula2
#p <- ggplot(df, aes(x, y))
p <- ggplot()+
  #four next lines of the code adapted from http://stackoverflow.com/questions/10448752/adding-x-and-y-axis-labels-in-ggplot2
  scale_size_area() + 
  xlab("Observations") +
  ylab("Prices difference") +
  ggtitle("Prices differences: cap rates Vasicek model Interest rates CIR model - MLE estimation type - without outliers")+
  geom_point(data=df1,aes(x=x,y=df1$y1),alpha=2/10, shape=21, fill="blue", colour="black", size=5) +
  geom_point(data=df2,aes(x=x,y=df2$y2),alpha=2/10, shape=21, fill="green", colour="black", size=5) +
  geom_smooth(data=df1,aes(x=x,y=df1$y1),method = "lm", se = FALSE, formula = my.formula1, colour = "red") +
  geom_smooth(data=df2,aes(x=x,y=df2$y2),method = "lm", se = FALSE, formula = my.formula2, colour = "blue")

m1 <- lm(my.formula1, df1)
m2 <- lm(my.formula2, df2)

my.eq1 <- as.character(signif(as.polynomial(coef(m1)), 4))
my.eq2 <- as.character(signif(as.polynomial(coef(m2)), 4))
label.text1 <- paste(gsub("x", "~italic(x)", my.eq1, fixed = TRUE))
label.text2 <- paste(gsub("x", "~italic(x)", my.eq2, fixed = TRUE))

p + annotate(geom = "text", x = -Inf, y =max(prices_comparison_apartment4_without_outlier)/2 , label = label.text1,
             family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "red")+
  #             family = "serif", hjust = 0, parse = TRUE, size = 4)+
  
  annotate(geom = "text", x = -Inf, y =min(prices_comparison_apartment4_without_outlier)/4, label = label.text2, 
           family = "serif", hjust = 0, parse = TRUE, size = 4,colour = "blue")

############################################################
#########MEASURES OF GOODNESS OF FIT
############################################################


#We load two functions taken from: https://heuristically.wordpress.com/2013/07/12/calculate-rmse-and-mae-in-r-and-sas/
# Function that returns Root Mean Squared Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

#We compute two measures of goodness of fit with the two measures for each type of estimation:

###############
###With Outlier
###############

#1. The Root Mean Square Error
rmse_estimation_1_apartment_with_outlier=rmse(prices_comparison_apartment1_with_outlier)
rmse_estimation_2_apartment_with_outlier=rmse(prices_comparison_apartment2_with_outlier)
rmse_estimation_3_apartment_with_outlier=rmse(prices_comparison_apartment3_with_outlier)
rmse_estimation_4_apartment_with_outlier=rmse(prices_comparison_apartment4_with_outlier)

#2. The Average Absolute Error (Mean Absolute Error in R)
mae_estimation_1_apartment_with_outlier=mae(prices_comparison_apartment1_with_outlier)
mae_estimation_2_apartment_with_outlier=mae(prices_comparison_apartment2_with_outlier)
mae_estimation_3_apartment_with_outlier=mae(prices_comparison_apartment3_with_outlier)
mae_estimation_4_apartment_with_outlier=mae(prices_comparison_apartment4_with_outlier)

##We create a dataframe with the results
rmse_estimation_apartment_with_outlier=c(rmse_estimation_1_apartment_with_outlier,rmse_estimation_2_apartment_with_outlier,rmse_estimation_3_apartment_with_outlier,rmse_estimation_4_apartment_with_outlier)
mae_estimation_apartment_with_outlier=c(mae_estimation_1_apartment_with_outlier,mae_estimation_2_apartment_with_outlier,mae_estimation_3_apartment_with_outlier,mae_estimation_4_apartment_with_outlier)
rmse_prices_difference_with_outlier=rbind(rmse_estimation_apartment_with_outlier,rmse_estimation_industrial_with_outlier,rmse_estimation_retail_with_outlier,rmse_estimation_office_with_outlier)
mae_prices_difference_with_outlier=rbind(mae_estimation_apartment_with_outlier,mae_estimation_industrial_with_outlier,mae_estimation_retail_with_outlier,mae_estimation_office_with_outlier)

#the other property types need to be loaded to have the results for mae_estimation_industrial, mae_estimation_retail and mae_estimation_office

#we give names to the columns
colnames(rmse_prices_difference_with_outlier) <- c("Constant Interest Rates","Adapted Interest Rates - Average Cap Rates","Vasicek CIR OLS estimation","Vasicek CIR MLE estimation")
colnames(mae_prices_difference_with_outlier) <- c("Constant Interest Rates","Adapted Interest Rates - Average Cap Rates","Vasicek CIR OLS estimation","Vasicek CIR MLE estimation")
rownames(rmse_prices_difference_with_outlier)<- c("Apartment","Industrial","Retail","Office")
rownames(mae_prices_difference_with_outlier)<- c("Apartment","Industrial","Retail","Office")

###############
###No Outlier
###############

#1. The Root Mean Square Error
rmse_estimation_1_apartment_without_outlier=rmse(prices_comparison_apartment1_without_outlier)
rmse_estimation_2_apartment_without_outlier=rmse(prices_comparison_apartment2_without_outlier)
rmse_estimation_3_apartment_without_outlier=rmse(prices_comparison_apartment3_without_outlier)
rmse_estimation_4_apartment_without_outlier=rmse(prices_comparison_apartment4_without_outlier)

#2. The Average Absolute Error (Mean Absolute Error in R)
mae_estimation_1_apartment_without_outlier=mae(prices_comparison_apartment1_without_outlier)
mae_estimation_2_apartment_without_outlier=mae(prices_comparison_apartment2_without_outlier)
mae_estimation_3_apartment_without_outlier=mae(prices_comparison_apartment3_without_outlier)
mae_estimation_4_apartment_without_outlier=mae(prices_comparison_apartment4_without_outlier)

##We create a dataframe without the outliers
rmse_estimation_apartment_without_outlier=c(rmse_estimation_1_apartment_without_outlier,rmse_estimation_2_apartment_without_outlier,rmse_estimation_3_apartment_without_outlier,rmse_estimation_4_apartment_without_outlier)
mae_estimation_apartment_without_outlier=c(mae_estimation_1_apartment_without_outlier,mae_estimation_2_apartment_without_outlier,mae_estimation_3_apartment_without_outlier,mae_estimation_4_apartment_without_outlier)
rmse_prices_difference_without_outlier=rbind(rmse_estimation_apartment_without_outlier,rmse_estimation_industrial_without_outlier,rmse_estimation_retail_without_outlier,rmse_estimation_office_without_outlier)
mae_prices_difference_without_outlier=rbind(mae_estimation_apartment_without_outlier,mae_estimation_industrial_without_outlier,mae_estimation_retail_without_outlier,mae_estimation_office_without_outlier)

#the other property types need to be loaded to have the results for mae_estimation_industrial, mae_estimation_retail and mae_estimation_office
#we give names to the columns
colnames(rmse_prices_difference_without_outlier) <- c("Constant Interest Rates","Adapted Interest Rates - Average Cap Rates","Vasicek CIR OLS estimation","Vasicek CIR MLE estimation")
colnames(mae_prices_difference_without_outlier) <- c("Constant Interest Rates","Adapted Interest Rates - Average Cap Rates","Vasicek CIR OLS estimation","Vasicek CIR MLE estimation")
rownames(rmse_prices_difference_without_outlier)<- c("Apartment","Industrial","Retail","Office")
rownames(mae_prices_difference_without_outlier)<- c("Apartment","Industrial","Retail","Office")

###############
###Benchmark
###############

#We provide the measures for the benchmark
rmse_estimation_apartment_benchmark=rmse(prices_comparison_benchmark_apartment)
rmse_estimation_industrial_benchmark=rmse(prices_comparison_benchmark_industrial)
rmse_estimation_retail_benchmark=rmse(prices_comparison_benchmark_retail)
rmse_estimation_office_benchmark=rmse(prices_comparison_benchmark_office)

mae_estimation_apartment_benchmark=mae(prices_comparison_benchmark_apartment)
mae_estimation_industrial_benchmark=mae(prices_comparison_benchmark_industrial)
mae_estimation_retail_benchmark=mae(prices_comparison_benchmark_retail)
mae_estimation_office_benchmark=mae(prices_comparison_benchmark_office)

#we give names to the columns
rmse_estimation_benchmark=c(rmse_estimation_apartment_benchmark,rmse_estimation_industrial_benchmark,rmse_estimation_retail_benchmark,rmse_estimation_office_benchmark)
ame_estimation_benchmark=c(mae_estimation_apartment_benchmark,mae_estimation_industrial_benchmark,mae_estimation_retail_benchmark,mae_estimation_office_benchmark)
measures_goodness_of_fit_benchmark=rbind(rmse_estimation_benchmark,ame_estimation_benchmark)
colnames(measures_goodness_of_fit_benchmark) <- c("Apartment","Industrial","Retail","Office")
rownames(measures_goodness_of_fit_benchmark)<- c("RMSE","AAE")

##############################################################
###############KOLMOGOROV SMIRNOV TEST########################
##############################################################

#Tests if the distributions are log normal
KS_test_distribution_log_normal=cbind(test_net_sales_prices_log_normal_dist_apartment_p_value,test_net_sales_prices_log_normal_dist_industrial_p_value,test_net_sales_prices_log_normal_dist_retail_p_value,test_net_sales_prices_log_normal_dist_office_p_value)
colnames(KS_test_distribution_log_normal) <-c("Apartment","Industrial","Retail","Office")
rownames(KS_test_distribution_log_normal) <- c("Tests log normal distributions")

#Simulations
#We test how similar the distributions of the Net Sale Prices and the Simulated Prices are.
KS_test_p_value_apartment1=ks.test(apartment_NetSalePrice,average_asset_price_apartment1)$p.value
KS_test_p_value_apartment2=ks.test(apartment_NetSalePrice,average_asset_price_apartment2)$p.value
KS_test_p_value_apartment3=ks.test(apartment_NetSalePrice,average_asset_price_apartment3)$p.value
KS_test_p_value_apartment4=ks.test(apartment_NetSalePrice,average_asset_price_apartment4)$p.value

#we create a table for the results
KS_test_p_value_apartment=c(KS_test_p_value_apartment1,KS_test_p_value_apartment2,KS_test_p_value_apartment3,KS_test_p_value_apartment4)
KS_test_p_value_industrial=c(KS_test_p_value_industrial1,KS_test_p_value_industrial2,KS_test_p_value_industrial3,KS_test_p_value_industrial4)
KS_test_p_value_retail=c(KS_test_p_value_retail1,KS_test_p_value_retail2,KS_test_p_value_retail3,KS_test_p_value_retail4)
KS_test_p_value_office=c(KS_test_p_value_office1,KS_test_p_value_office2,KS_test_p_value_office3,KS_test_p_value_office4)
KS_test=rbind(KS_test_p_value_apartment,KS_test_p_value_industrial,KS_test_p_value_retail,KS_test_p_value_office)
colnames(KS_test) <- c("Constant Interest Rates","Adapted Interest Rates - Average Cap Rates","Vasicek CIR OLS estimation","Vasicek CIR MLE estimation")
rownames(KS_test) <- c("KS_test_p_value_apartment","KS_test_p_value_industrial","KS_test_p_value_retail","KS_test_p_value_office")

#Benchmark
#we provide a similar table for the benchmark
KS_test_p_value_apartment_benchmark=ks.test(apartment_NetSalePrice,prices_comparison_benchmark_apartment)$p.value
KS_test_p_value_industrial_benchmark=ks.test(industrial_NetSalePrice,prices_comparison_benchmark_industrial)$p.value
KS_test_p_value_retail_benchmark=ks.test(retail_NetSalePrice,prices_comparison_benchmark_retail)$p.value
KS_test_p_value_office_benchmark=ks.test(office_NetSalePrice,prices_comparison_benchmark_office)$p.value
KS_test_benchmark=cbind(KS_test_p_value_apartment_benchmark,KS_test_p_value_industrial_benchmark,KS_test_p_value_retail_benchmark,KS_test_p_value_office_benchmark)
colnames(KS_test_benchmark) <- c("Apartment_benchmark","Industrial_benchmark","Retail_benchmark","Office_benchmark")

##############################################################
###############SUMMARY OF ALL COEFFICIENTS####################
#####################OLS/ MLE ESTIMATION######################

#####VASICEK

###Coefficents a, b and sigma - OLS
a_hat_cap_rates_ols
b_hat_cap_rates_ols
sigma_hat_cap_rates_ols

###Coefficents a, b and sigma - MLE
a_hat_cap_rates_mle=theta1_Vasicek_MLE
b_hat_cap_rates_mle=theta2_Vasicek_MLE/a_hat_cap_rates_mle
sigma_hat_cap_rates_mle=theta3_Vasicek_MLE

#average for the coefficients of the cap rates
a_hat_cap_rates_ols_average_apartment=mean(a_hat_cap_rates_ols)
b_hat_cap_rates_ols_average_apartment=mean(b_hat_cap_rates_ols)
sigma_hat_cap_rates_ols_average_apartment=mean(sigma_hat_cap_rates_ols)

a_hat_cap_rates_mle_average_apartment=mean(a_hat_cap_rates_mle)
b_hat_cap_rates_mle_average_apartment=mean(b_hat_cap_rates_mle)
sigma_hat_cap_rates_mle_average_apartment=mean(sigma_hat_cap_rates_mle)

#We create a dataframe with the variables a,b and sigma in order to export it to Latex
coefficients_from_sitmo=c(3.1288,0.9075,0.5831)
#taken from http://www.sitmo.com/article/calibrating-the-ornstein-uhlenbeck-model/

#example of coefficients from Amin page 33 (see references thesis for more information about Amin)
coefficients_Amin=c(0.3906,0.0238,0.0325)

#have been taken from Sharp, Netwon and Duck 2008
#An improved Fixed Rate Mortgage valuation of Methodology with Interacting Prepayment and Default Options
coefficients_Reference_Article=c(0.25, 0.10,0.05)
hat_cap_rates_ols_apartment=c(a_hat_cap_rates_ols_average_apartment,b_hat_cap_rates_ols_average_apartment,sigma_hat_cap_rates_ols_average_apartment)
hat_cap_rates_mle_apartment=c(a_hat_cap_rates_mle_average_apartment,b_hat_cap_rates_mle_average_apartment,sigma_hat_cap_rates_mle_average_apartment)

#we run the code for industrial, office and retail and gather all the results in one table
parameters_estimated=rbind(hat_cap_rates_ols_apartment,hat_cap_rates_mle_apartment,hat_cap_rates_ols_industrial,hat_cap_rates_mle_industrial,hat_cap_rates_ols_retail,hat_cap_rates_mle_retail,hat_cap_rates_ols_office,hat_cap_rates_mle_office,coefficients_from_sitmo,coefficients_Amin)
colnames(parameters_estimated) <- c("kappa_hat","theta_hat","sigma_hat")

#we make a distinction between the coefficients obtained from Vasicek and the ones from CIR
#coefficients CIR - they are the same for apartment, industrial, retail or office.
a_hat_interest_rates_mle=theta1_CIR_MLE_2
b_hat_interest_rates_mle=theta2_CIR_MLE_2/a_hat_interest_rates_mle
sigma_hat_interest_rates_mle=theta3_CIR_MLE_2

a_hat_interest_rates_mle_apartment=a_hat_interest_rates_mle
b_hat_interest_rates_mle_apartment=b_hat_interest_rates_mle
sigma_hat_interest_rates_mle_apartment=sigma_hat_interest_rates_mle

hat_interest_rates_OLS_apartment=c(a_hat_interest_rates_OLS,b_hat_interest_rates_OLS,sigma_hat_interest_rates_OLS)
hat_interest_rates_MLE_apartment=c(a_hat_interest_rates_mle,b_hat_interest_rates_mle,sigma_hat_interest_rates_mle_apartment)
coefficients_CIR_Iacus=c(0.9194592, 0.1654958/0.9194592, 0.8255179)

#we create the table
coefficients_interest_rates=rbind(hat_interest_rates_OLS_apartment,hat_interest_rates_MLE_apartment,coefficients_CIR_Iacus)
colnames(coefficients_interest_rates) <- c("kappa_hat","theta_hat","sigma_hat")

##############################################################
#######COMPUTATION OF THE NUMBER OF TIMES#####################
#######OUR SIMULATIONS OUTPERFORM THE BENCHMARK###############

#we initialize the counting variables
nb_better_than_benchmark_apartment1=0
nb_better_than_benchmark_apartment2=0
nb_better_than_benchmark_apartment3=0
nb_better_than_benchmark_apartment4=0
for (i in 1:length(prices_comparison_apartment1)){
  if (abs(prices_comparison_apartment1)[i] < abs(prices_comparison_benchmark_apartment)[i]){
    nb_better_than_benchmark_apartment1=nb_better_than_benchmark_apartment1+1
    }
}

for (i in 1:length(prices_comparison_apartment2)){
  if (abs(prices_comparison_apartment2)[i] < abs(prices_comparison_benchmark_apartment)[i]){
    nb_better_than_benchmark_apartment2=nb_better_than_benchmark_apartment2+1
  }
}

for (i in 1:length(prices_comparison_apartment3)){
  if (abs(prices_comparison_apartment3)[i] < abs(prices_comparison_benchmark_apartment)[i]){
    nb_better_than_benchmark_apartment3=nb_better_than_benchmark_apartment3+1
  }
}

for (i in 1:length(prices_comparison_apartment4)){
  if (abs(prices_comparison_apartment4)[i] < abs(prices_comparison_benchmark_apartment)[i]){
    nb_better_than_benchmark_apartment4=nb_better_than_benchmark_apartment4+1
  }
}
  
#we create a table with the results
simulations_outperform_benchmark_apartment=c(nb_better_than_benchmark_apartment1,nb_better_than_benchmark_apartment2,nb_better_than_benchmark_apartment3,nb_better_than_benchmark_apartment4)
simulations_outperform_benchmark_industrial=c(nb_better_than_benchmark_industrial1,nb_better_than_benchmark_industrial2,nb_better_than_benchmark_industrial3,nb_better_than_benchmark_industrial4)
simulations_outperform_benchmark_retail=c(nb_better_than_benchmark_retail1,nb_better_than_benchmark_retail2,nb_better_than_benchmark_retail3,nb_better_than_benchmark_retail4)
simulations_outperform_benchmark_office=c(nb_better_than_benchmark_office1,nb_better_than_benchmark_office2,nb_better_than_benchmark_office3,nb_better_than_benchmark_office4)
outperformance_of_the_benchmark=rbind(simulations_outperform_benchmark_apartment,simulations_outperform_benchmark_industrial,simulations_outperform_benchmark_retail,simulations_outperform_benchmark_office)
colnames(outperformance_of_the_benchmark) <- c("Constant Interest Rates","Adapted Interest Rates - Average Cap Rates","Vasicek CIR OLS estimation","Vasicek CIR MLE estimation")
rownames(outperformance_of_the_benchmark) <- c("Apartment","Industrial","Retail","Office")

##############################################################
#######NUMBER OF OUTLIERS REMOVED#############################
##############################################################
nb_outliers_removed_apartment=c(nb_outliers_removed_OLS_estimation_apartment_prices_comparison_apartment1,nb_outliers_removed_OLS_estimation_apartment_prices_comparison_apartment2,nb_outliers_removed_OLS_estimation_apartment_prices_comparison_apartment3,nb_outliers_removed_OLS_estimation_apartment_prices_comparison_apartment4)                                               
nb_outliers_removed_industrial=c(nb_outliers_removed_OLS_estimation_industrial_prices_comparison_industrial1,nb_outliers_removed_OLS_estimation_industrial_prices_comparison_industrial2,nb_outliers_removed_OLS_estimation_industrial_prices_comparison_industrial3,nb_outliers_removed_OLS_estimation_industrial_prices_comparison_industrial4)                                               
nb_outliers_removed_retail=c(nb_outliers_removed_OLS_estimation_retail_prices_comparison_retail1,nb_outliers_removed_OLS_estimation_retail_prices_comparison_retail2,nb_outliers_removed_OLS_estimation_retail_prices_comparison_retail3,nb_outliers_removed_OLS_estimation_retail_prices_comparison_retail4)                                               
nb_outliers_removed_office=c(nb_outliers_removed_OLS_estimation_office_prices_comparison_office1,nb_outliers_removed_OLS_estimation_office_prices_comparison_office2,nb_outliers_removed_OLS_estimation_office_prices_comparison_office3,nb_outliers_removed_OLS_estimation_office_prices_comparison_office4)                                               
nb_outliers_removed=rbind(nb_outliers_removed_apartment,nb_outliers_removed_industrial,nb_outliers_removed_retail,nb_outliers_removed_office)
colnames(nb_outliers_removed) <- c("Constant Interest Rates","Adapted Interest Rates - Average Cap Rates","Vasicek CIR OLS estimation","Vasicek CIR MLE estimation")
rownames(nb_outliers_removed) <- c("Apartment","Industrial","Retail","Office")

##############################################################
#######ANALYSIS OF THE PRICES DISTRIBUTION####################
##############################################################

plot(density(prices_comparison_apartment1))
prices_test=threshold(prices_comparison_apartment1, min = -1.0e+08, max = 1.0e+08)
hist(prices_test)
x=seq(-1.0e+05,max = 1.0e+05)
dnorm(x, mean = mean(prices_test), sd = sd(prices_test))
var(prices_test)
plot(density(dnorm(x, mean = mean(prices_test), sd = sd(prices_test))),add=TRUE)

##############################################################
###############EXPORTATION OF THE DATA########################
####################TO LATEX##################################

#table of the test of log normal distributions
xtable(KS_test_distribution_log_normal)

#table for the rmse with outlier
xtable(rmse_prices_difference_with_outlier)
#table for the mae with outlier
xtable(mae_prices_difference_with_outlier)
#table for the rmse without outlier
xtable(rmse_prices_difference_without_outlier)
#table for the mae without outlier
xtable(mae_prices_difference_without_outlier)

#table for the measures of goodness of fit benchmark
xtable(measures_goodness_of_fit_benchmark)

#cap rates
xtable(parameters_estimated)

#table for the KS tests - simulations
xtable(KS_test)
xtableMatharray(KS_test,digits=6)
#table for the KS test - benchmark
xtable(KS_test_benchmark)
xtableMatharray(KS_test_benchmark,digits=6)

#table outperformance of the benchmark
xtable(outperformance_of_the_benchmark)

#table for the coefficients of the interest rates
xtable(coefficients_interest_rates)

#table for the number of outliers removed
xtable(nb_outliers_removed)

##Export the file into a PDF
knitr::stitch("Code_12_Mai_Apartment_LB_69.r")

################################END#########################################
############################################################################
############################################################################
