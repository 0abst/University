% III. TESTING ASSET PRICING MODELS

% III. 1) Capital Asset Pricing Model (CAPM)

%% Data Importation - Industry Portfolios - Book to Market - French
%% Research factors
industry_portfolio = dataset('XLSFile','10IndustryPortfoliosfichierxls.xlsx');
data_industry_portfolio=xlsread('10IndustryPortfoliosfichierxls.xlsx');
booktomarket=dataset('XLSFile','25booktoMarketandSizePortfolios.xlsx');
data_booktomarket=xlsread('25booktoMarketandSizePortfolios.xlsx');
French_research_factors = dataset('XLSFile','French research factors.xlsx');
data_French_research_factors = xlsread('French research factors.xlsx');

% a) Estimation of the CAPM equations on the excess returns - focus on the
% 10 Industry Portfolios

% We convert the first column (date) into a date format:
Date1=datenum(num2str(data_industry_portfolio(:,1)),'yyyymm');

%  We create matrixes/vectors for our parameters of interest
Returns_industry_portfolio=data_industry_portfolio(:,2:11);

% As we have two more values for the French factors, we remove the two last
% values

Mkt_Rf=data_French_research_factors(1:end,2);
SMB=data_French_research_factors(1:end,3);
HML=data_French_research_factors(1:end,4);
Rf=data_French_research_factors(1:end,5);

% We find the size of the matrixes
size_Returns_industry_portfolio=size(Returns_industry_portfolio);
nb_values=size_Returns_industry_portfolio(1,1);
nb_columns=size_Returns_industry_portfolio(1,2);

%For directly getting the results of the regressions, we use the Lesage
%Toolbox:
% Loading the Lesage Toolbox:
addpath(genpath('XXXX'));

%  OLS estimate:
%rho1_betas=ones(1:2,nb_columns);
%rho1_t_stat=ones(1:2,nb_columns);
%rho1_std=ones(1:2,nb_columns);
%rho1_rsqr=ones(nb_columns);

for i=1:nb_columns;
      Y=Returns_industry_portfolio(:,i) - Rf;
      X=[ones(nb_values,1), Mkt_Rf];
      rho1(i)=ols(Y,X);
      rho1_betas(1:2,i)=rho1(i).beta;
      rho1_t_stat(1:2,i)=rho1(i).tstat;
      rho1_std(1:2,i)=rho1(i).bstd;
      rho1_rsqr(i)=rho1(i).rsqr;
end

%Returns predicted by CAPM
%We use the values from OLS
%We compute the mean of Y

% CAPM model - we assume that a=0

for i=1:nb_columns;
    X_CAPM_Returns_industry_portfolio(:,i)=rho1_betas(2,i)*Mkt_Rf;
    mean_X_CAPM_Returns_industry_portfolio(i)=mean(X_CAPM_Returns(:,i));
    mean_Y(i)=mean(Returns_industry_portfolio(:,i) - Rf);
end ;   

%b) Plot of mean excess returns versus mean excess returns predicted by
%CAPM
scatter(12*mean_X_CAPM_Returns_industry_portfolio,12*mean_Y,'marker','o')
title('US industry portfolios');
xlabel('Predicted mean excess returns');
ylabel('Mean excess returns');

% c)Estimate CAPM equations on the excess returns for each out of 25 book
% to market and size portfolios.

addpath(genpath('XXXX'));

% We convert the first column (date) into a date format:
Date2=datenum(num2str(data_booktomarket(:,1)),'yyyymm');

%  We create matrixes/vectors for our parameters of interest
Returns_booktomarket=data_booktomarket(:,2:26);

% We set up the size of the matrixes
size_Returns_booktomarket=size(Returns_booktomarket);
nb_values=size_Returns_booktomarket(1,1);
nb_columns=size_Returns_booktomarket(1,2);

%For directly getting the results of the regressions, we use the Lesage
%Toolbox:
% Loading the Lesage Toolbox:
addpath(genpath('XXXX'));

%  OLS estimate:
%rho1_betas=ones(1:2,nb_columns);
%rho1_t_stat=ones(1:2,nb_columns);
%rho1_std=ones(1:2,nb_columns);
%rho1_rsqr=ones(nb_columns);

for i=1:nb_columns;
      Y=Returns_booktomarket(:,i) - Rf;
      X=[ones(nb_values,1), Mkt_Rf];
      rho2(i)=ols(Y,X);
      rho2_betas(1:2,i)=rho2(i).beta;
      rho2_t_stat(1:2,i)=rho2(i).tstat;
      rho2_std(1:2,i)=rho2(i).bstd;
      rho2_rsqr(i)=rho2(i).rsqr;
end

%Returns predicted by CAPM
%We use the values from OLS
%We compute the mean of Y

% CAPM model - hypothesis H0: alpha=0

for i=1:nb_columns;
    X_CAPM_Returns_booktomarket(:,i)=rho2_betas(2,i)*Mkt_Rf;
    mean_X_CAPM_Returns_booktomarket(i)=mean(X_CAPM_Returns_booktomarket(:,i));
    mean_Y(i)=mean(Returns_booktomarket(:,i) - Rf);
end ;   

% Plot of mean excess returns versus mean excess returns predicted by
%CAPM
scatter(12*mean_X_CAPM_Returns_booktomarket,12*mean_Y)
title('US book-to-market amd size portfolios');
xlabel('Predicted mean excess returns');
ylabel('Mean excess returns');
axis([0 15 0 15])


% 3.2 Fama French Model

% Industry Portfolios

% We convert the first column (date) into a date format:
Date3=datenum(num2str(data_industry_portfolio(:,1)),'yyyymm');

%  We create matrixes/vectors for our parameters of interest
Returns_industry_portfolio=data_industry_portfolio(:,2:11);
Returns_booktomarket=data_booktomarket(:,2:26);

% We set up the size of the matrixes
size_Returns_industry_portfolio=size(Returns_industry_portfolio);
nb_values=size_Returns_industry_portfolio(1,1);
nb_columns=size_Returns_industry_portfolio(1,2);

%For directly getting the results of the regressions, we use the Lesage
%Toolbox:
% Loading the Lesage Toolbox:
addpath(genpath('XXXX'));

%  OLS estimate:
%rho1_betas=ones(1:2,nb_columns);
%rho1_t_stat=ones(1:2,nb_columns);
%rho1_std=ones(1:2,nb_columns);
%rho1_rsqr=ones(nb_columns);

for i=1:nb_columns;
      Y=Returns_industry_portfolio(:,i) - Rf;
      X=[ones(nb_values,1), Mkt_Rf, SMB, HML];
      rho3(i)=ols(Y,X);
      rho3_betas(1:4,i)=rho3(i).beta;
      rho3_t_stat(1:4,i)=rho3(i).tstat;
      rho3_std(1:4,i)=rho3(i).bstd;
      rho3_rsqr(i)=rho3(i).rsqr;
end

%Returns predicted by CAPM
%We use the values from OLS
%We compute the mean of Y

% CAPM model - we assume that a=0

for i=1:nb_columns;
    X_CAPM_Returns_industry_portfolio(:,i)=rho3_betas(2,i)*Mkt_Rf + rho3_betas(3,i)*SMB+rho3_betas(4,i)*HML;
    mean_X_CAPM_Returns_industry_portfolio(i)=mean(X_CAPM_Returns_industry_portfolio(:,i));
    mean_Y(i)=mean(Returns_industry_portfolio(:,i) - Rf);
end ;   

% Plot of mean excess returns versus mean excess returns predicted by
%CAPM
scatter(12*mean_X_CAPM_Returns_industry_portfolio,12*mean_Y)
title('US book-to-market amd size portfolios');
xlabel('Predicted mean excess returns');
ylabel('Mean excess returns');


% 25 Portfolios

addpath(genpath('XXXX'));

% We convert the first column (date) into a date format:
Date4=datenum(num2str(data_booktomarket(:,1)),'yyyymm');

%  We create matrixes/vectors for our parameters of interest
Returns_booktomarket=data_booktomarket(:,2:26);

% We set up the size of the matrixes
size_Returns_booktomarket=size(Returns_booktomarket);
nb_values=size_Returns_booktomarket(1,1);
nb_columns=size_Returns_booktomarket(1,2);

%For directly getting the results of the regressions, we use the Lesage
%Toolbox:
% Loading the Lesage Toolbox:
addpath(genpath('XXXX'));

%  OLS estimate:
%rho1_betas=ones(1:2,nb_columns);
%rho1_t_stat=ones(1:2,nb_columns);
%rho1_std=ones(1:2,nb_columns);
%rho1_rsqr=ones(nb_columns);

for i=1:nb_columns;
      Y=Returns_booktomarket(:,i) - Rf;
      X=[ones(nb_values,1), Mkt_Rf, SMB, HML];
      rho4(i)=ols(Y,X);
      rho4_betas(1:4,i)=rho4(i).beta;
      rho4_t_stat(1:4,i)=rho4(i).tstat;
      rho4_std(1:4,i)=rho4(i).bstd;
      rho4_rsqr(i)=rho4(i).rsqr;
end

%Returns predicted by CAPM
%We use the values from OLS
%We compute the mean of Y

% CAPM model - we assume that a=0

for i=1:nb_columns;
    X_CAPM_Returns_booktomarket(:,i)=rho4_betas(2,i)*Mkt_Rf + rho4_betas(3,i)*SMB+rho4_betas(4,i)*HML;
    mean_X_CAPM_Returns_booktomarket(i)=mean(X_CAPM_Returns_booktomarket(:,i));
    mean_Y(i)=mean(Returns_booktomarket(:,i) - Rf);
end ;   

% Plot of mean excess returns versus mean excess returns predicted by
%CAPM
scatter(12*mean_X_CAPM_Returns_booktomarket,12*mean_Y)
title('US book-to-market amd size portfolios');
xlabel('Predicted mean excess returns');
ylabel('Mean excess returns');


