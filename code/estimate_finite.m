clear

% Load simulated data
data = importdata('data/data_finite.csv');

%Setting the starting values and bounds for the optimisation
startval = [14;0.0001;0.4;0.4];
lb = [0,0,0,0];
ub = [10000,1,1,1];

%Optimising the log-likelihood to estimate the paramaters
opt = optimset('TolFun',1E-10,'TolX',1E-10,'MaxFunEvals',1000,'Display','iter');
[x,~,~,~,~,grad,hessian] = fmincon(@(pars)rust_loglik_finite(data,pars), startval, [], ...
[], [], [], lb, ub, [], opt);

%Computing standard errors and confidence intervals
fisher = inv(hessian);
se = sqrt(diag(fisher));
lowerconf = -(1.96 .* se) + x;
upperconf = (1.96 .* se) + x;

% old code:
% opt = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter','StepTolerance',1E-14,'MaxFunctionEvaluations',4000);
% fminunc(@(pars)rust_loglik(data,pars,1e-6),startval,opt)

