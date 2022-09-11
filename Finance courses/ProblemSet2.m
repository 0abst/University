%% To see the dataset/ Change the Path in File/Set path
stocks = dataset('XLSFile','PS1.xls');

%% Data Importation
data=xlsread('PS1.xls');
libor=importdata('1m_CHF_Libor_Rates.csv');
libor_average=mean(libor.data);

%% II. RISK MEASURES

%% 2.1 COMPARING DIFFERENT RISK MEASURES

% SET UP OF THE DATA

%a)We consider ABB and CSGN for the global period and two periods.
data=xlsread('PS1.xls');
stocks1 = dataset('XLSFile','PS1.xls');
data2=xlsread('PS2.xlsx');
stocks2 = dataset('XLSFile','PS2.xlsx');
data3=xlsread('PS3.xlsx');
stocks3 = dataset('XLSFile','PS3.xlsx');


% We consider the returns for ABB, ATLN, ADEN and CSGN

%Global period
ABB1=stocks1.ABB;
CSGN1=stocks1.CSGN;

%First period Jan 2004 - Dec 2007
ABB2=stocks2.ABB;
CSGN2=stocks2.CSGN;

%Second period Jan 2008 - Dec 2014
ABB3=stocks3.ABB;
CSGN3=stocks3.CSGN;

Returns_ABB1=(ABB1(2:end,:)-ABB1(1:end-1,:))./ABB1(1:end-1,:);
Returns_CSGN1=(CSGN1(2:end,:)-CSGN1(1:end-1,:))./CSGN1(1:end-1,:);
Returns_ABB2=(ABB2(2:end,:)-ABB2(1:end-1,:))./ABB2(1:end-1,:);
Returns_CSGN2=(CSGN2(2:end,:)-CSGN2(1:end-1,:))./CSGN2(1:end-1,:);
Returns_ABB3=(ABB3(2:end,:)-ABB3(1:end-1,:))./ABB3(1:end-1,:);
Returns_CSGN3=(CSGN3(2:end,:)-CSGN3(1:end-1,:))./CSGN3(1:end-1,:);

%%Risk measures

%Standard deviation
%Global Period
std_ABB1=std(Returns_ABB1);
std_CSGN1=std(Returns_CSGN1);

%First period Jan 2004 - Dec 2007
std_ABB2=std(Returns_ABB2);
std_CSGN2=std(Returns_CSGN2);

%Second period Jan 2008 - Dec 2014
std_ABB3=std(Returns_ABB3);
std_CSGN3=std(Returns_CSGN3);

%Mean absolute deviation
%Global Period
mad_ABB1=mad(Returns_ABB1);
mad_CSGN1=mad(Returns_CSGN1);

%First period Jan 2004 - Dec 2007
mad_ABB2=mad(Returns_ABB2);
mad_CSGN2=mad(Returns_CSGN2);

%Second period Jan 2008 - Dec 2014
mad_ABB3=mad(Returns_ABB3);
mad_CSGN3=mad(Returns_CSGN3);

%Semi standard deviation
%In measuring risk, the standard deviation will calculate the overall dispersion around a mean or expected value with disregard for the proportion of the downside and upside dispersions. On the other hand, the semideviation quantifies the magnitude of the downside and upside deviations solely.
%cf: http://ch.mathworks.com/matlabcentral/fileexchange/45251-semideviation

%let us create a function for this purpose - the file semistd need to be in
%the same repertory as the present file

lower_semi_std_dev_ABB_1=semistd(Returns_ABB1);
lower_semi_std_dev_CSGN_1=semistd(Returns_CSGN1);
lower_semi_std_dev_ABB_2=semistd(Returns_ABB2);
lower_semi_std_dev_CSGN_2=semistd(Returns_CSGN2);
lower_semi_std_dev_ABB_=semistd(Returns_ABB2);
lower_semi_std_dev_CSGN_3=semistd(Returns_CSGN3);

%% VAR

%We first need to compute the average returns mu:
mu_ABB1=mean(Returns_ABB1);
mu_ABB2=mean(Returns_ABB2);
mu_ABB3=mean(Returns_ABB3);
mu_CSGN1=mean(Returns_CSGN1);
mu_CSGN2=mean(Returns_CSGN2);
mu_CSGN3=mean(Returns_CSGN3);

%95% conf level
%We need to compute the quantiles

c_95 = norminv([0.05 0.95],0,1);

VAR_ABB1_95=-(mu_ABB1+c_95(1,1)*std_ABB1);
VAR_ABB2_95=-(mu_ABB2+c_95(1,1)*std_ABB2);
VAR_ABB3_95=-(mu_ABB3+c_95(1,1)*std_ABB3);
VAR_CSGN1_95=-(mu_CSGN1+c_95(1,1)*std_CSGN1);
VAR_CSGN2_95=-(mu_CSGN2+c_95(1,1)*std_CSGN2);
VAR_CSGN3_95=-(mu_CSGN3+c_95(1,1)*std_CSGN3);

%99% conf level

% Compute of the quantiles for 99%
c_99 = norminv([0.01 0.99],0,1);

VAR_ABB1_99=-(mu_ABB1+c_99(1,1)*std_ABB1);
VAR_ABB2_99=-(mu_ABB2+c_99(1,1)*std_ABB2);
VAR_ABB3_99=-(mu_ABB3+c_99(1,1)*std_ABB3);
VAR_CSGN1_99=-(mu_CSGN1+c_99(1,1)*std_CSGN1);
VAR_CSGN2_99=-(mu_CSGN2+c_99(1,1)*std_CSGN2);
VAR_CSGN3_99=-(mu_CSGN3+c_99(1,1)*std_CSGN3);

%% Expected shortfall:
%We implement the formula of lecture 6 page 13

%Expected shortfall - 95%
ES_ABB1_95=-mu_ABB1+normpdf(c_95(1,1))*std_ABB1/(1-95/100);
ES_ABB2_95=-mu_ABB2+normpdf(c_95(1,1))*std_ABB2/0.05;
ES_ABB3_95=-mu_ABB3+normpdf(c_95(1,1))*std_ABB3/0.05;
ES_CSGN1_95=-mu_CSGN1+normpdf(c_95(1,1))*std_CSGN1/0.05;
ES_CSGN2_95=-mu_CSGN2+normpdf(c_95(1,1))*std_CSGN2/0.05;
ES_CSGN3_95=-mu_CSGN3+normpdf(c_95(1,1))*std_CSGN3/0.05;

%Expected shortfall - 99%
ES_ABB1_99=-mu_ABB1+normpdf(c_99(1,1))*std_ABB1/0.01;
ES_ABB2_99=-mu_ABB2+normpdf(c_99(1,1))*std_ABB2/0.01;
ES_ABB3_99=-mu_ABB3+normpdf(c_99(1,1))*std_ABB3/0.01;
ES_CSGN1_99=-mu_CSGN1+normpdf(c_99(1,1))*std_CSGN1/0.01;
ES_CSGN2_99=-mu_CSGN2+normpdf(c_99(1,1))*std_CSGN2/0.01;
ES_CSGN3_99=-mu_CSGN3+normpdf(c_99(1,1))*std_CSGN3/0.01;

%%%%%%%%%%%END FOR THIS PART

% b) QQplot
% Empirical quantiles
grid_size=99;
ligne=linspace(0.01,0.99,grid_size);
emp_quantiles=quantile(Returns_CSGN3,ligne);
%recall: values for mu and variance for this stock - this is used for
%drawing the normal dist
mu_CSGN3=mean(Returns_CSGN3);
std_CSGN3=std(Returns_CSGN3);
var_CSGN3=std_CSGN3^2;
cum_dist_norm= normpdf(Returns_CSGN3,mu_CSGN3,std_CSGN3);
%regarder si on doit mettre la var ou sigma (internet -> sigma)
quantiles_from_estimated_N=quantile(cum_dist_norm,ligne);
%QQ plot
qqplot(quantiles_from_estimated_N,emp_quantiles);
xlabel('Quantiles from estimated N(mu,sigma^2)') ;
ylabel('Emirical quantiles') ;
title('QQ plot of weekly Credit Suisse Returns Jan 08 Dec 14');

X = norminv(ligne,mu_CSGN3,std_CSGN3);

qqplot(X,emp_quantiles);
xlabel('Quantiles from estimated N(mu,sigma^2)') ;
ylabel('Emirical quantiles') ;
title('QQ plot of weekly Credit Suisse Returns Jan 08 Dec 14');

%c) No assumption of the distribution
VaR95_no_assum_ABB = -[quantile(Returns_ABB1,1-0.95);quantile(Returns_ABB2,1-0.95);quantile(Returns_ABB3,1-0.95)]*100;
VaR95_no_assum_CSGN = -[quantile(Returns_CSGN1,1-0.95);quantile(Returns_CSGN2,1-0.95);quantile(Returns_CSGN3,1-0.95)]*100;

VaR99_no_assum_ABB = -[quantile(Returns_ABB1,1-0.99);quantile(Returns_ABB2,1-0.99);quantile(Returns_ABB3,1-0.99)]*100;
VaR99_no_assum_CSGN = -[quantile(Returns_CSGN1,1-0.99);quantile(Returns_CSGN2,1-0.99);quantile(Returns_CSGN3,1-0.99)]*100;

% %Extra 20 points - Bootstrapping
% 
% %Rng not supported by my Matlab version (too old)/ so I used these commands
% %instead of
% s = RandStream('mt19937ar','Seed',100);
% RandStream.setDefaultStream(s);
% 
% New_std_CSGN_boot=bootstrp(1000,@mean,Returns_CSGN1);
% standard_deviation=std(New_std_CSGN_boot)
% hist(New_std_CSGN_boot)
% 
% y = datasample(Returns_CSGN1,1000);
% z = bootstrp(1000,@std,Returns_CSGN1);

%  PLEASE CHECK THE PYTHON CODE FOR THE BOOTSTRAP


%% 2.2 BACKTESTING

% we compare the actual returns in the first week of jan 2008 
% to the VaR and we see what happens.
VAR_ABB1_99=-(mu_ABB1+c_99(1,1)*std_ABB1);
Returns_ABB2(1);
%  For ABB2, we are above the VAR
VAR_CSGN_99=-(mu_CSGN1+c_99(1,1)*std_CSGN1);
Returns_CSGN2(1);
%  For Credit Suisse, we are below the VAR

%%Now we extend the estimation window by includind the first week of Jan
%%2008. So we compute new values of the VAR. We need to compute new values
%%of the mu and the std.

countABB = 0
countCSGN = 0
for i=1:length(Returns_ABB3)-1;
    new_mean_ABB(i)= mean(Returns_ABB1(1:207+i));
    new_std_ABB(i) = std(Returns_ABB1(1:207+i));
    new_VaR_ABB(i) = -(new_mean_ABB(i)+c_99(1,1)*new_std_ABB(i));
    if -new_VaR_ABB(i) > Returns_ABB1(208+i)
        countABB = countABB + 1
    end
    
    new_mean_CSGN(i)= mean(Returns_CSGN1(1:207+i));
    new_std_CSGN(i) = std(Returns_CSGN1(1:207+i));
    new_VaR_CSGN(i) = -(new_mean_CSGN(i)+c_99(1,1)*new_std_CSGN(i));
    if -new_VaR_CSGN(i) > Returns_CSGN1(208+i)
        countCSGN = countCSGN + 1
    end
end
