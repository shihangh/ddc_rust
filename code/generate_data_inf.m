clear

% Initialise parameter values
% Parameter values taken from col 2 of table 9 in Rust (1987)
%beta = 0.999;               % discount factor, could be smaller to speed up convergence
%p_x0 = 0.3919;              % probability of making [0-5000) miles
%p_x1 = 0.5953;              % probability of making [5000-10000) miles
%p_x2 = 1- p_x0 - p_x1;      % probability of making [10000 - \infty) miles    
    
pars = [10.0750;0.00005293;0.3919;0.5953];   % Starting values for data generation
rc = pars(1);           % Replacement cost
theta1_1 = pars(2);     % Maintenance cost paramater with a linear cost function
p_x0 = pars(3);         % Transition probs
p_x1 = pars(4);

beta = 0.8;             % Discount factor
p_x2 = 1- p_x0 - p_x1;
it_tol = 1e-6;         % tolerance paramater to define minimum distance in contraction mapping algorithm

x_grid = transpose(0:5000:350000); %Set up discretised state space of x

% Vector of costs in each period, for each choice
u_1 = arrayfun(@(x) cost(1,x,pars),x_grid); % Compute u(a,x) for a = 1 (71 times 1)
u_0 = arrayfun(@(x) cost(0,x,pars),x_grid); % Compute u(a,x) for a = 0 (71 times 1)

% Create matrix Fx_0 (71 times 71)
% Remember that this is because you may transition from any state to any
% other state, but the probabilities of transitioning to most states are set to zero, by assumption.
Fx_0 = zeros(length(x_grid),length(x_grid));
for i = 1:length(x_grid)
    Fx_0(i,i) = p_x0;
    if i <= length(x_grid) - 1
        Fx_0(i,i+1) = p_x1;
    end
    if i <= length(x_grid) - 2
        Fx_0(i,i+2) = p_x2;
    end
end
Fx_0(length(x_grid)-1,length(x_grid)) = 1- p_x0;
Fx_0(length(x_grid),length(x_grid)) = 1;

% Create matrix Fx_1 (71 times 71)
Fx_1row = zeros(1,length(x_grid));
Fx_1row(1) = p_x0;
Fx_1row(2) = p_x1;
Fx_1row(3) = p_x2;
Fx_1 = repmat(Fx_1row,length(x_grid),1);

% Iteration
Vbar = inner_algo(Fx_1,Fx_0,u_0,u_1,beta,1e-6);

% Testing code - not necessary%
% Vbar_0 = log(exp(u_0) + exp(u_1));
% Vbar_1 = iteration(Vbar_0,Fx_1,Fx_0,u_0,u_1,beta);
% Vbar_2 = iteration(Vbar_1,Fx_1,Fx_0,u_0,u_1,beta);

% Write the v(a,x)s
v_0 = u_0 + (beta .* (Fx_0 * Vbar));
v_1 = u_1 + (beta .* (Fx_1 * Vbar));

exp_v0 = exp(v_0);
exp_v1 = exp(v_1);
sum_v = exp_v0 + exp_v1;
choiceprob_1 = exp_v1 ./ sum_v;
choiceprob_0 = exp_v0 ./ sum_v;

% Simulate the dataset
rng(1)
N = 1000;
T = 24;
    % Generate a set of uniform random variables which we will transform
choicerand = rand(N,T);
transitionrand = rand(N,T);
    
    % States
statemat = NaN(N,T);
        % Start at 0 in t = 1
statemat(:,1) = 0;
choicemat = NaN(N,T);
evmat = NaN(N,T);
xnextmat = NaN(N,T);
    % Draw the evs (regardless of a)
for i = 1:N
    for j = 1:T
        if transitionrand(i,j) <= p_x0
            evmat(i,j) = 0;
        elseif transitionrand(i,j) > p_x0 && transitionrand(i,j) <= p_x0+p_x1
            evmat(i,j) = 5000;
        elseif transitionrand(i,j) > p_x0+p_x1
            evmat(i,j) = 10000;
        end
    end
end

    % Fill in the rest
for i = 1:N
    for j = 1:T
        if j > 1
            statemat(i,j) = xnextmat(i,j-1);
        end
        statematindex = floor(statemat(i,j) ./5000)+1;
            % Fill in the choices based on x
        if choicerand(i,j) <= choiceprob_1(statematindex)
            choicemat(i,j) = 1;
        elseif choicerand(i,j) > choiceprob_1(statematindex)
            choicemat(i,j) = 0;
        end
            % Fill in next period x
        if choicemat(i,j) == 1 
            xnextmat(i,j) = evmat(i,j);
        elseif choicemat(i,j) == 0
            xnextmat(i,j) = statemat(i,j) + evmat(i,j);
        end
    end
end

% Reshape data
statedat = statemat(:,5:17);   % We arbiritrarly pick the observations between the 5th and 17th period
statedat = reshape(transpose(statedat),[],1);
choicedat = choicemat(:,5:17);
choicedat = reshape(transpose(choicedat),[],1);
xnextdat = xnextmat(:,5:17);
xnextdat = reshape(transpose(xnextdat),[],1);


data = [choicedat statedat xnextdat];
writematrix(data,'data/data.csv')