clear

% Load simulated data
data = importdata('data/data.csv');

% Set starting values and bounds for the optimisation algorithm
startval = [5;0.001;0.4;0.4];
lb = [0,0,0,0];
ub = [10000,1,1,1];

% Estimate paramaters by optimising the log-likelihood function
opt = optimset('TolFun',1E-20,'TolX',1E-20,'MaxFunEvals',1000,'Display','iter');
[x,~,~,~,~,grad,hessian] = fmincon(@(pars)rust_loglik_inf(data,pars,1e-10), startval, [], ...
[], [], [], lb, ub, [], opt);

% Estimate standard errors and confidence intervals for the paramaters
fisher = inv(hessian);
se = sqrt(diag(fisher));
lowerconf = -(1.96 .* se) + x;
upperconf = (1.96 .* se) + x;

% This was old code, to test whether the inner and outer algorithms were
% working. No need to use it.
%
% pars = [10.0750;0.00005293;0.3919;0.5953];
% rust_loglik(data,pars,1e-6)
% 
% opt = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter','StepTolerance',1E-14,'MaxFunctionEvaluations',4000);
% fminunc(@(pars)rust_loglik(data,pars,1e-6),startval,opt)