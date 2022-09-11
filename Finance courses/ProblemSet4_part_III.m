% We load the set for the predictors
Russel_dataset=dataset('XLSFile','Russel_index.xlsx');
Russel=xlsread('Russel_index.xlsx');

% predictors_dataset=dataset('XLSFile','database_problem_set_4.xlsx');
predictors=xlsread('database_problem_set_4.xlsx');

%Path for the Lesage Toolbox: we add the path
addpath(genpath('XXXX'));

%  we create a matrix X_y1 for the predicors - period 1
for i=2:3913;
    for j=1:8;
    X_y1(i-1,j)=predictors(i,j);
    end;
end;

%  we create a matrix X_y2 for the predicors - period 2 - moving window
for i=3913:4107; % we are one period before because of the lag
    for j=1:8;
    X_y2(i-3912,j)=predictors(i,j);
    end;
end;

%  Russell
% we are one step forward for Russel compared to the Predictors - we load
% the data
y1=Russel(2:3913);
y2=Russel(3914:4108);

% we create a matrix of ones - needed for the regression
x1=1:1:3912;
ones_x1=x1';

% we get R^2 for the in-sample
pred_y1=[ones_x1 X_y1];
rho=ols(y1,pred_y1);
R_squared_in_sample=rho.rsqr;

% Benchmark model predictor - we compute the average of y1
average_russel=mean(y1);

% We initialize the value
sum_y2=0;
% Mean for the benchmark
for i=1:194;
    sum_y2(i+1)=sum_y2(i)+y2(i);
    new_mean_y2(i)=sum_y2(i)/i;
%      new_mean_y2_global(i)=[mean(y1) new_mean_y2(i)];
end;
%  Global vector we have
new_mean_y2_global=[mean(y1) new_mean_y2];
new_mean_y2_global=new_mean_y2_global';

% we compute the ols
for i=1:194;
%     for j=1:8;
    X_y1_y2=[X_y1(i:end,:) ; X_y2(1:i,:)];
    Y_y1_y2=[y1(i:end,:);y2(1:i)];
    x2=i:1:(3912+i);
    ones_x2=x2';
    pred_y1=[ones_x2 X_y1_y2];
    rho_y1_y2=ols(Y_y1_y2,X_y1_y2);
end

% The betas we have:
rho_y1_y2.beta

% The t-values we have:
rho_y1_y2.tstat

% For the first forecast, we take the value of the 31th december 2014 to
% forecast the value of the 1st January
Russel_first_predicted_value=X_y1(end,:)*rho_y1_y2.beta;

% Let us compute the returns we have 
for i=1:194
    returns_Russel_forecasted(i)=X_y2(i,:)*rho_y1_y2.beta;
end
% All returns forecasted are then
all_returns_Russel_forecasted=[Russel_first_predicted_value  returns_Russel_forecasted];
all_returns_Russel_forecasted=all_returns_Russel_forecasted';

% we plot the data
scatter(all_returns_Russel_forecasted,y2);
title('Jan—Sep 2015: Plot of the return forecasts by the best model vs actual values of the Russell 3000');
xlabel('Return forecasts by the Best Model');
ylabel('Actual values of the Russell 3000');

%  We compute the number of values for the second period (195)
nb_values=size(y2);
nb_values=nb_values(1,1);

% Compute of the out of sample R^2
% sum((y2-all_returns_Russel_forecasted)^2)
for i=1:195
    diff_predicted(i)=(y2(i)-all_returns_Russel_forecasted(i))^2;
    diff_benchmark(i)=(y2(i)-new_mean_y2_global(i))^2;
end

diff_predicted=diff_predicted';
diff_benchmark=diff_benchmark';

%  we take the sum
sum_diff_predicted=sum(diff_predicted);
sum_diff_benchmark=sum(diff_benchmark);

% We divide by the number of values
sum_diff_predicted_nb_values=sum_diff_predicted/nb_values;
sum_diff_benchmark_nb_values=sum_diff_benchmark/nb_values;

% Final Compute of hte R^2 out of sample
R_squared_out_of_sample=1-sum_diff_predicted_nb_values/sum_diff_benchmark_nb_values;
