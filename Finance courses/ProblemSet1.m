%% To see the dataset/ Change the Path in File/Set path
stocks = dataset('XLSFile','PS1.xls');

%% Data Importation
data=xlsread('PS1.xls');
libor=importdata('1m_CHF_Libor_Rates.csv');
libor_average=mean(libor.data);

%% I. PORTFOLIO CHOICE AND MEAN VARIANCE FRONTIER

%% 1.1 DIVERSIFICATION AND LEVERAGE

%a)

%%VOIR SI IMPORTANT
%Count of the number of values
n=size(stocks,1);
%Count of the number of stocks
s_data=size(data);
nb_stocks=s_data(1,2);

%%%%% A VOIR:Count of hte number of years
n_years=floor(n/52);

%%Compute of the Arithmetic Returns for the 18 stocks -two ways
%For loop
Ari_Returns=ones(n,nb_stocks);
for j=1:nb_stocks;
for i=2:n;
Ari_Returns(i,j)=data(i,j)/data(i-1,j)-1;
end;
end;
%Directly
Ari_Returns_2=(data(2:end,:)-data(1:end-1,:))./data(1:end-1,:);

%We keep the direct result (Ari_Return_2) for the sequel

%%Computation of the Average annualized return, standard deviation and
%%Sharpe Ratio SR

    %Average annualized return:
    Average_annualized_return=52*mean(Ari_Returns_2);
    Average_annualized_return_2=Average_annualized_return';
    %Standard deviation
    Average_annualized_Standard_deviation=sqrt(52)*std(Ari_Returns_2);
    %Sharpe Ratio
    Average_annualized_Sharpe_ratio=Average_annualized_return./Average_annualized_Standard_deviation;

%%%What are the average annualized mu, sigma and SR accross the 18 stocks?    
    Average_annualized_mu=mean(Average_annualized_return)
    Average_annualized_sigma=mean(Average_annualized_Standard_deviation)
    Average_annualized_SR=mean(Average_annualized_Sharpe_ratio)
    
%b)Enter diversification
%Equally weighted portfolio

eq_w_port=1/nb_stocks*ones(nb_stocks,1);
Sigma_Average_Returns=cov(Ari_Returns_2);

Ari_Returns_2_eq_w_port=eq_w_port'*Ari_Returns_2';
Sigma_Ari_Returns_2=eq_w_port'*Sigma_Average_Returns*eq_w_port;

%Average annualized return - equally weighted portfolio:
    Average_annualized_return_eq_w=52*mean(Ari_Returns_2_eq_w_port)
    %Standard deviation
    Average_annualized_Standard_deviation_eq_w=sqrt(52)*std(Ari_Returns_2_eq_w_port)
    %Sharpe Ratio
    Average_annualized_Sharpe_ratio_eq_w=(Average_annualized_return_eq_w-libor_average)./Average_annualized_Standard_deviation_eq_w

%%%What are the average annualized mu, sigma and SR accross the 18 stocks - eqully weighted portfolio?    
    Average_annualized_mu=mean(Average_annualized_return_eq_w);
    Average_annualized_sigma=mean(Average_annualized_Standard_deviation_eq_w);
    Average_annualized_SR=mean(Average_annualized_Sharpe_ratio_eq_w);
    
%% c)Enter leverage: we take the results from part a)

%Average annualized return - equally weighted portfolio - we take the
%variables from above
    Average_annualized_return_eq_w=52*mean(Ari_Returns_2_eq_w_port);
    %Standard deviation
    Average_annualized_Standard_deviation_eq_w=sqrt(52)*std(Ari_Returns_2_eq_w_port);
    %Sharpe Ratio
    Average_annualized_Sharpe_ratio_eq_w=(Average_annualized_return_eq_w-libor_average/100)./Average_annualized_Standard_deviation_eq_w;

%%%What are the average annualized mu, sigma and SR accross the 18 stocks - eqully weighted portfolio - we just renames our variables    
    Average_annualized_mu=Average_annualized_return_eq_w;
    Average_annualized_sigma=Average_annualized_Standard_deviation_eq_w;
    Average_annualized_SR=Average_annualized_Sharpe_ratio_eq_w;

%Leverage Ratio of 1: the risk free rate is not taken into account. Same
%results as in a)
    %Average annualized return:
    Average_annualized_return=Average_annualized_mu;
    %Standard deviation
    Average_annualized_Standard_deviation=Average_annualized_sigma;
    %Sharpe Ratio
    Average_annualized_Sharpe_ratio=Average_annualized_Sharpe_ratio_eq_w;

%Leverage Ratio of 3:
v=3;
    %Average annualized return:
    Average_annualized_return_lev_3=v*Average_annualized_mu+(1-v)*libor_average/100;
    %Standard deviation
    Average_annualized_Standard_deviation_lev_3=v*Average_annualized_sigma;
    %Sharpe Ratio
    Average_annualized_Sharpe_ratio_lev_3=(Average_annualized_return_lev_3-libor_average/100)./(v*Average_annualized_sigma);

%Leverage Ratio of 5:
v=5;
    Average_annualized_return_lev_3=v*Average_annualized_mu+(1-v)*libor_average/100;
    %Standard deviation
    Average_annualized_Standard_deviation_lev_3=v*Average_annualized_sigma;
    %Sharpe Ratio
    Average_annualized_Sharpe_ratio_lev_3=(Average_annualized_return_lev_3-libor_average/100)./(v*Average_annualized_sigma);
    
%% 1.2 MEAN-VARIANCE FRONTIER AND OPTIMAL PORTFOLIO

% We take two stocks: Credit Suisse and Zurich Insurance
%%1.1 Construction of the mean-variance frontier

%a)
CSGN=stocks.CSGN;
ZURN=stocks.ZURN;   

%We compute the annualized returns:
Returns_CSGN=(CSGN(2:end,:)-CSGN(1:end-1,:))./CSGN(1:end-1,:);
Returns_ZURN=(ZURN(2:end,:)-ZURN(1:end-1,:))./ZURN(1:end-1,:);

Annualized_returns_CSGN=52*mean(Returns_CSGN);
Annualized_returns_ZURN=52*mean(Returns_ZURN);

%%We have four cases depending on the correlation between the assets
% Case (a): Actual correlation between the assets

Returns_mu=[Annualized_returns_CSGN;Annualized_returns_ZURN];
Sigma_Act_cov=cov(Returns_CSGN,Returns_ZURN);

%Creation of the grid       
grid_size=11;
weights=[linspace(0,1,grid_size);linspace(1,0,grid_size)];
Returns_mv_front_Act_cov=ones(grid_size,1);
Sigma_Act_cov_mv_front=ones(grid_size,1);

for i=1:grid_size;
    Returns_mv_front_Act_cov(i)=weights(:,i)'*Returns_mu;
    Sigma_Act_cov_mv_front(i)=sqrt(weights(:,i)'*Sigma_Act_cov*weights(:,i))*sqrt(52);
end;

plot(Sigma_Act_cov_mv_front,Returns_mv_front_Act_cov);

% Case (b): correlation of -1 between the assets

Returns_mu=[Annualized_returns_CSGN;Annualized_returns_ZURN];
Sigma_perf_neg=[var(Returns_CSGN), -std(Returns_CSGN)*std(Returns_ZURN);-std(Returns_CSGN)*std(Returns_ZURN),var(Returns_ZURN)];

grid_size=11
weights=[linspace(0,1,grid_size);linspace(1,0,grid_size)]
Returns_mv_front_perf_neg_corr=ones(grid_size,1)
Sigma_perf_neg_corr_mv_front=ones(grid_size,1)

for i=1:grid_size;
    Returns_mv_front_perf_neg_corr(i)=weights(:,i)'*Returns_mu;
    Sigma_perf_neg_corr_mv_front(i)=sqrt(weights(:,i)'*Sigma_perf_neg*weights(:,i))*sqrt(52);
end;

plot(abs(Sigma_perf_neg_corr_mv_front),Returns_mv_front_perf_neg_corr);

% Case (c): correlation of 0 between the assets

Returns_mu=[Annualized_returns_CSGN;Annualized_returns_ZURN];
Sigma_zero_corr=[var(Returns_CSGN), 0;0,var(Returns_ZURN)];

grid_size=11;
weights=[linspace(0,1,grid_size);linspace(1,0,grid_size)];
Returns_mv_front_zero_corr=ones(grid_size,1);
Sigma_perf_zero_corr_mv_front=ones(grid_size,1);

for i=1:grid_size;
    Returns_mv_front_zero_corr(i)=weights(:,i)'*Returns_mu;
    Sigma_perf_zero_corr_mv_front(i)=sqrt(weights(:,i)'*Sigma_zero_corr*weights(:,i))*sqrt(52);
end;

plot(Sigma_perf_zero_corr_mv_front,Returns_mv_front_zero_corr);

% Case (d): correlation of 1 between the assets

Returns_mu=[Annualized_returns_CSGN;Annualized_returns_ZURN];
Sigma_perf_pos=[var(Returns_CSGN), std(Returns_CSGN)*std(Returns_ZURN);std(Returns_CSGN)*std(Returns_ZURN),var(Returns_ZURN)];

grid_size=11;
weights=[linspace(0,1,grid_size);linspace(1,0,grid_size)];
Returns_mv_front_perf_pos_corr=ones(grid_size,1);
Sigma_perf_pos_corr_mv_front=ones(grid_size,1);

for i=1:grid_size;
    Returns_mv_front_perf_pos_corr(i)=weights(:,i)'*Returns_mu;
    Sigma_perf_pos_corr_mv_front(i)=sqrt(weights(:,i)'*Sigma_perf_pos*weights(:,i))*sqrt(52);
end;

plot(Sigma_perf_pos_corr_mv_front,Returns_mv_front_perf_pos_corr, 'r');

%%%%%We plot all graphs on the same cover

plot(Sigma_Act_cov_mv_front,Returns_mv_front_Act_cov,'y')
hold on
plot(abs(Sigma_perf_neg_corr_mv_front),Returns_mv_front_perf_neg_corr,'g')
hold on
plot(Sigma_perf_zero_corr_mv_front,Returns_mv_front_zero_corr,'b')
hold on
plot(Sigma_perf_pos_corr_mv_front,Returns_mv_front_perf_pos_corr, 'r')
legend('corr=actual','corr=-1','corr=0','corr=1','Location','SouthWest')
title('Mean variance frontier of Credit Suisse and Zurich Insurance with different correlations')


% b)Let us assume that short sales are allowed. We need to find an optimal
% portfolio that offers the minimum variance

%%We allow for negative weights - we go from -2 to 1 and from 1 to -2.
% We consider the case for which we have the actual covariance
Returns_mu=[Annualized_returns_CSGN;Annualized_returns_ZURN];
Sigma_Act_cov=cov(Returns_CSGN,Returns_ZURN);

%Creation of the grid       
grid_size=31;
weights_short_sales=[linspace(-2,1,grid_size);linspace(2,1,grid_size)];
Returns_mv_front_Act_cov_short_sales=ones(grid_size,1);
Sigma_Act_cov_mv_front_short_sales=ones(grid_size,1);

for i=1:grid_size-1;
    Returns_mv_front_Act_cov_short_sales(i)=weights_short_sales(:,i)'*Returns_mu;
    Sigma_Act_cov_mv_front_short_sales(i)=sqrt(weights_short_sales(:,i)'*Sigma_Act_cov*weights_short_sales(:,i))*sqrt(52);
         if Sigma_Act_cov_mv_front_short_sales(i+1) < Sigma_Act_cov_mv_front_short_sales(i);
             Min_variance=Sigma_Act_cov_mv_front_short_sales(i+1);
             weight_with_minimum_variance=weights_short_sales(:,i);
         end;
end

plot(Sigma_Act_cov_mv_front_short_sales,Returns_mv_front_Act_cov_short_sales)
legend('corr=actual','corr=-1','corr=0','corr=1','Location','SouthWest')
title('Mean variance frontier of Credit Suisse and Zurich Insurance for actual correlations - short sales allowed')

%% We plot the mean variance frontier when short sales are allowed and not

plot(Sigma_Act_cov_mv_front,Returns_mv_front_Act_cov,'r')
hold on
plot(Sigma_Act_cov_mv_front_short_sales,Returns_mv_front_Act_cov_short_sales)
legend('short sales not allowed','short sales allowed','Location','SouthEast')
title('Mean variance frontier of Credit Suisse and Zurich Insurance for actual correlations - short sales allowed')


%% III. MULTI-ASSET MVF AND TANGENCY PORTFOLIO

% a) We consider the returns for ABB, ATLN, ADEN and CSGN

ABB=stocks.ABB;
ATLN=stocks.ATLN;   
ADEN=stocks.ADEN;
CSGN=stocks.CSGN;

%We compute the returns:
Returns_ABB=(ABB(2:end,:)-ABB(1:end-1,:))./ABB(1:end-1,:);
Returns_ATLN=(ATLN(2:end,:)-ATLN(1:end-1,:))./ATLN(1:end-1,:);
Returns_ADEN=(ADEN(2:end,:)-ADEN(1:end-1,:))./ADEN(1:end-1,:);
Returns_CSGN=(CSGN(2:end,:)-CSGN(1:end-1,:))./CSGN(1:end-1,:);

% We compute the annualized retunrs
Annualized_returns_ABB=52*mean(Returns_ABB);
Annualized_returns_ATLN=52*mean(Returns_ATLN);
Annualized_returns_ADEN=52*mean(Returns_ADEN);
Annualized_returns_CSGN=52*mean(Returns_CSGN);

%%We create percentages
Annualized_returns_ABB_per=100*52*mean(Returns_ABB);
Annualized_returns_ATLN_per=100*52*mean(Returns_ATLN);
Annualized_returns_ADEN_per=100*52*mean(Returns_ADEN);
Annualized_returns_CSGN_per=100*52*mean(Returns_CSGN);

%% Inputs

%%Covariance matrix of asset returns
mu_Returns=[Annualized_returns_ABB_per,Annualized_returns_ATLN_per,Annualized_returns_ADEN_per,Annualized_returns_CSGN_per];
Returns=[Returns_ABB,Returns_ATLN,Returns_ADEN,Returns_CSGN];
Sigma=cov(Returns);
Sigma_inv=inv(Sigma);

%Libor
libor_average=mean(libor.data);

%Expected returns
mu_star=(0:1:39)';
%Mean excess returns
mu_e=mu_Returns-libor_average;

%%Calculating the weights
w=ones(4,40);
sum_weights=ones(1,40);
weight_Risk_Free_Rate=ones(1,40);

for i=1:40;
w(:,i)=(mu_star(i)-libor_average)./(mu_e*Sigma_inv*mu_e')*Sigma_inv*mu_e';
sum_weights(1,i)=sum(w(:,i));
end;
weight_Risk_Free_Rate=1-sum_weights;
Weight_risky_asset_1st_portfolio=weight_Risk_Free_Rate(1);
Weight_risky_asset_20st_portfolio=weight_Risk_Free_Rate(20);
Weight_risky_asset_40st_portfolio=weight_Risk_Free_Rate(40);

%% b) For each of the forty portfolios compute the standard deviation:

Standard_deviation_portfolio=ones(1,40);
Returns_portfolio=ones(1,40);
for i=1:40;
w(:,i)=(mu_star(i)-libor_average)./(mu_e*Sigma_inv*mu_e')*Sigma_inv*mu_e';
sum_weights(1,i)=sum(w(:,i));
Standard_deviation_portfolio(1,i)=sqrt(w(:,i)'*Sigma*w(:,i))*sqrt(52);
end;

% We plot the mean variance frontier
plot(Standard_deviation_portfolio,mu_star)
xlabel('Standard deviation of return') ;
ylabel('Expected return') ;
title('Mean-Variance Frontier');
ylim([0 10])

%% c) We plot the mean variance frontier for the 18 stocks

%Compute of the returns:

Returns_18_stocks=zeros(574,18);
Annualized_returns_18_stocks=zeros(1,18);
Annualized_returns_18_stocks_per=zeros(1,18);
for j=1:18;
    for i=2:574;
        Returns_18_stocks(i,j)=(data(i,j)/data(i-1,j))-1;
    end;
    Annualized_returns_18_stocks_per(1,j)=100*52*mean(Returns_18_stocks([2:574],j));   
end;

% Returns_18_stocks(:,1)
% Annualized_returns_18_stocks_per(:,1)

%% Inputs

%%Covariance matrix of asset returns
mu_Returns_18_stocks=[Annualized_returns_18_stocks_per];
Returns=[Returns_18_stocks];
Sigma_18_stocks=cov(Returns_18_stocks);
Sigma_inv_18_stocks=inv(Sigma_18_stocks);

%Expected returns
mu_star_18_stocks=(0:1:39)';

%Mean excess returns - No investment in the risk free asset
mu_e_18_stocks=mu_Returns_18_stocks;

%%Calculating the weights
weights_18_stocks=ones(18,40);

%Standard deviation
Standard_deviation_portfolio_18_stocks=ones(1,40);

for i=1:40;
weights_18_stocks(:,i)=(mu_star_18_stocks(i)-libor_average)./(mu_e_18_stocks*Sigma_inv_18_stocks*mu_e_18_stocks')*Sigma_inv_18_stocks*mu_e_18_stocks';
Standard_deviation_portfolio_18_stocks(1,i)=sqrt(weights_18_stocks(:,i)'*Sigma_18_stocks*weights_18_stocks(:,i))*sqrt(52);
end;

% We plot the mean variance frontier
plot(Standard_deviation_portfolio_18_stocks,mu_star_18_stocks, 'g')
hold on
plot(Standard_deviation_portfolio,mu_star, 'b')
xlabel('Standard deviation of return') ;
ylabel('Expected return %') ;
title('Mean-Variance Frontier');
legend('Multi-asset MVF 18 stocks','Multi-asset MVF 4 stocks','Location','SouthEast');
ylim([0 10])

%% d) Weights of risky assets in the tangency portfolio:

% Tangency portfolio weights:
tangency_portfolio_weights_18_stocks=Sigma_inv_18_stocks*mu_e_18_stocks'./((ones(18,1)')*Sigma_inv_18_stocks*mu_e_18_stocks');

weight_ABB_tangency_portfolio=tangency_portfolio_weights_18_stocks(1)

% We check if the sum equals 1
sum(tangency_portfolio_weights_18_stocks)
% Indeed we find that the sum equals 1

%%%Need of ploting the tangency portfolio
Mean_Returns_portfolio_18_stocks_tangency_portfolio=tangency_portfolio_weights_18_stocks'*mu_e_18_stocks';
Standard_deviation_tangency_portfolio_18_stocks=sqrt(52)*sqrt(tangency_portfolio_weights_18_stocks'*Sigma_18_stocks*tangency_portfolio_weights_18_stocks);

% As we know that tangency portfolio crosses the mean axes when the
% standard deviation equals 0, we know that b=Rf
% Consequently, we do:
% y = a*x + b => a = (y-b)/x

% We find for a: 
slope=(Mean_Returns_portfolio_18_stocks_tangency_portfolio-libor_average)./Standard_deviation_tangency_portfolio_18_stocks

x=[0:1/100:40/100]
size(x)
y=slope*x+libor_average

plot(x,y)

% Plot

plot(Standard_deviation_portfolio_18_stocks,mu_star_18_stocks,'r')
hold on
plot(x,y,'b')
xlabel('Standard deviation of return %') ;
ylabel('Expected return %') ;
title('Mean-Variance Frontier');
legend('Multi-asset MVF 18 stocks','Tangency portfolio','Location','SouthEast')
ylim([0 10])



%% Extra 15 points

%we create two other files - PS2 and PS3 which split the data. The
%PS2 covers the period from Jan 2004 to Dec 2007. The PS3 covers
%the period Jan 2008 to Dec 2014. We give the names data1 and data2 to the
%data.


% Data Importation

addpath(genpath('C:\Users\Loic\Documents\HSG\Semestre 3\Theory of Finance\Assignemnt 1\Assignment 1\Matlab code'));

data1=xlsread('PS2.xls');
stocks1 = dataset('XLSFile','PS2.xls');
data2=xlsread('PS3.xls');
stocks2 = dataset('XLSFile','PS3.xls');

%% PERIOD JAN 2004 - DEC 2007 (DATA 1)

% a) We consider the returns for ABB, ATLN, ADEN and CSGN

ABB=stocks1.ABB;
ATLN=stocks1.ATLN;   
ADEN=stocks1.ADEN;
CSGN=stocks1.CSGN;

%We compute the returns:
Returns_ABB=(ABB(2:end,:)-ABB(1:end-1,:))./ABB(1:end-1,:);
Returns_ATLN=(ATLN(2:end,:)-ATLN(1:end-1,:))./ATLN(1:end-1,:);
Returns_ADEN=(ADEN(2:end,:)-ADEN(1:end-1,:))./ADEN(1:end-1,:);
Returns_CSGN=(CSGN(2:end,:)-CSGN(1:end-1,:))./CSGN(1:end-1,:);

% We compute the annualized retunrs
Annualized_returns_ABB=52*mean(Returns_ABB);
Annualized_returns_ATLN=52*mean(Returns_ATLN);
Annualized_returns_ADEN=52*mean(Returns_ADEN);
Annualized_returns_CSGN=52*mean(Returns_CSGN);

%%We create percentages
Annualized_returns_ABB_per=100*52*mean(Returns_ABB);
Annualized_returns_ATLN_per=100*52*mean(Returns_ATLN);
Annualized_returns_ADEN_per=100*52*mean(Returns_ADEN);
Annualized_returns_CSGN_per=100*52*mean(Returns_CSGN);

%% Inputs

%%Covariance matrix of asset returns
mu_Returns=[Annualized_returns_ABB_per,Annualized_returns_ATLN_per,Annualized_returns_ADEN_per,Annualized_returns_CSGN_per];
Returns=[Returns_ABB,Returns_ATLN,Returns_ADEN,Returns_CSGN];
Sigma=cov(Returns);
Sigma_inv=inv(Sigma);

%Libor
libor_average=mean(libor.data);

%Expected returns
mu_star2=(0:1:39)';
%Mean excess returns
mu_e=mu_Returns-libor_average;

%%Calculating the weights
w=ones(4,40);
sum_weights=ones(1,40);
weight_Risk_Free_Rate=ones(1,40);

for i=1:40;
w(:,i)=(mu_star(i)-libor_average)./(mu_e*Sigma_inv*mu_e')*Sigma_inv*mu_e';
sum_weights(1,i)=sum(w(:,i));
end;

%% b) For each of the forty portfolios compute the standard deviation:

Standard_deviation_portfolio2=ones(1,40);
Returns_portfolio=ones(1,40);
for i=1:40;
w(:,i)=(mu_star2(i)-libor_average)./(mu_e*Sigma_inv*mu_e')*Sigma_inv*mu_e';
sum_weights(1,i)=sum(w(:,i));
Standard_deviation_portfolio2(1,i)=sqrt(w(:,i)'*Sigma*w(:,i))*sqrt(52);
end;

% We plot the mean variance frontier
plot(Standard_deviation_portfolio2,mu_star2)
xlabel('Standard deviation of return %') ;
ylabel('Expected return %') ;
title('Mean-Variance Frontier');
ylim([0 10])

%%Weights of ABB in the 0% and 40% return portfolio - period: Jan 2004 - Dec 2007:
weight_ABB_1st_port_data1=w(1,1)
weight_ABB_40th_port_data1=w(1,40)

%% PERIOD JAN 2008 - DEC 2014 (DATA 2)

% a) We consider the returns for ABB, ATLN, ADEN and CSGN

ABB=stocks2.ABB;
ATLN=stocks2.ATLN;   
ADEN=stocks2.ADEN;
CSGN=stocks2.CSGN;

%We compute the returns:
Returns_ABB=(ABB(2:end,:)-ABB(1:end-1,:))./ABB(1:end-1,:);
Returns_ATLN=(ATLN(2:end,:)-ATLN(1:end-1,:))./ATLN(1:end-1,:);
Returns_ADEN=(ADEN(2:end,:)-ADEN(1:end-1,:))./ADEN(1:end-1,:);
Returns_CSGN=(CSGN(2:end,:)-CSGN(1:end-1,:))./CSGN(1:end-1,:);

% We compute the annualized retunrs
Annualized_returns_ABB=52*mean(Returns_ABB);
Annualized_returns_ATLN=52*mean(Returns_ATLN);
Annualized_returns_ADEN=52*mean(Returns_ADEN);
Annualized_returns_CSGN=52*mean(Returns_CSGN);

%%We create percentages
Annualized_returns_ABB_per=100*52*mean(Returns_ABB);
Annualized_returns_ATLN_per=100*52*mean(Returns_ATLN);
Annualized_returns_ADEN_per=100*52*mean(Returns_ADEN);
Annualized_returns_CSGN_per=100*52*mean(Returns_CSGN);

%% Inputs

%%Covariance matrix of asset returns
mu_Returns=[Annualized_returns_ABB_per,Annualized_returns_ATLN_per,Annualized_returns_ADEN_per,Annualized_returns_CSGN_per];
Returns=[Returns_ABB,Returns_ATLN,Returns_ADEN,Returns_CSGN];
Sigma=cov(Returns);
Sigma_inv=inv(Sigma);

%Libor
libor_average=mean(libor.data);

%Expected returns
mu_star3=(0:1:39)';
%Mean excess returns
mu_e=mu_Returns-libor_average;

%%Calculating the weights
w=ones(4,40);
sum_weights=ones(1,40);
weight_Risk_Free_Rate=ones(1,40);

for i=1:40;
w(:,i)=(mu_star2(i)-libor_average)./(mu_e*Sigma_inv*mu_e')*Sigma_inv*mu_e';
sum_weights(1,i)=sum(w(:,i));
end;
weight_Risk_Free_Rate=1-sum_weights;
Weight_risky_asset_1st_portfolio=weight_Risk_Free_Rate(1);
Weight_risky_asset_20st_portfolio=weight_Risk_Free_Rate(20);
Weight_risky_asset_40st_portfolio=weight_Risk_Free_Rate(40);

%% b) For each of the forty portfolios compute the standard deviation:

Standard_deviation_portfolio3=ones(1,40);
Returns_portfolio=ones(1,40);
for i=1:40;
w(:,i)=(mu_star3(i)-libor_average)./(mu_e*Sigma_inv*mu_e')*Sigma_inv*mu_e';
sum_weights(1,i)=sum(w(:,i));
Standard_deviation_portfolio3(1,i)=sqrt(w(:,i)'*Sigma*w(:,i))*sqrt(52);
end;

% We plot the mean variance frontier
plot(Standard_deviation_portfolio2,mu_star2, 'b')
hold on
plot(Standard_deviation_portfolio3,mu_star3, 'r')
xlabel('Standard deviation of return') ;
ylabel('Expected return %') ;
title('Mean-Variance Frontier');
ylim([0 10])
legend('period: Jan 2004 - Dec 2007','period: Jan 2008 - Dec 2014','Location','SouthEast')
title('Mean variance frontier for the two sub-samples')

%%Weights of ABB in the 0% and 40% return portfolio - period: Jan 2008 - Dec 2014:
weight_ABB_1st_port_data2=w(1,1)
weight_ABB_40th_port_data2=w(1,40)


