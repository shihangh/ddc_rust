clear

% Initialise parameter values
    % Parameter values taken from col 2 of table 9 in Rust (1987)
%beta = 0.999;               % discount factor, could be smaller to speed up convergence
%p_x0 = 0.3919;              % probability of making [0-5000) miles
%p_x1 = 0.5953;              % probability of making [5000-10000) miles
%p_x2 = 1- p_x0 - p_x1;      % probability of making [10000 - \infty) miles    
    
%pars = [20; 0.5 ; 0; 0.6];
pars = [15.0750;0.0005293;0.3919;0.5953];
rc = pars(1);
theta1_1 = pars(2);
p_x0 = pars(3);
p_x1 = pars(4);

beta = 0.8;
p_x2 = 1- p_x0 - p_x1;
it_tol = 1e-6;         % tolerance paramater to define minimum distance in contraction mapping algorithm

N = 1000; %Set the number of simulated buses that will be created in the data
T = 12;  %Set the number of periods

x_grid = transpose(0:5000:350000); %Set up discretised state space of x
u_1 = arrayfun(@(x) cost(1,x,pars),x_grid); % Compute u(a,x) for a = 1 (71 times 1)
u_0 = arrayfun(@(x) cost(0,x,pars),x_grid); % Compute u(a,x) for a = 0 (71 times 1)

% Create matrix Fx_0 (71 times 71) -- this is the transition probability
% conditional on maintaing the current enging
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

% Create matrix Fx_1 (71 times 71)-- this is the transition probability
% matrix conditional on maintaing the replacing the enging (so it
% "regenerates" (i.e. resets to zero) the state variable after replacement)

Fx_1row = zeros(1,length(x_grid));
Fx_1row(1) = p_x0;
Fx_1row(2) = p_x1;
Fx_1row(3) = p_x2;
Fx_1 = repmat(Fx_1row,length(x_grid),1);

%%%%%% Backwards induction for finite discrete time

%Create a matrix of the value function for every state and every time
%period
V = NaN(length(x_grid),T);

%Set the terminal value, for all the state
V(:,T) = log(exp(u_1) + exp(u_0));

%Set intermediate values by backward induction, starting from the
%penultimate value until the first
for t = T-1:-1:1 
   V(:,t) = log(exp(u_1 + (beta .* (Fx_1 * V(:,t+1)))) + exp(u_0 + (beta .* (Fx_0 * V(:,t+1)))));
end

% Creating the value functions conditional on the choice to evaluate the
% choice probabilities at each state-time period.
V_0 = NaN(length(x_grid),T);
V_1 = NaN(length(x_grid),T);
V_0(:,T) = exp(u_0);
V_1(:,T) = exp(u_1);
% Backward induction using the utility from the period-specific choice, u_i, and the values computed
% earlier for the optimal path for the rest of the periods, V(S,T).
for t = T-1:-1:1
   V_0(:,t) = exp(u_0 + (beta .* (Fx_0 * V(:,t+1))));
   V_1(:,t) = exp(u_1 + (beta .* (Fx_1 * V(:,t+1))));
end

choiceprobs = V_1 ./ (V_0 + V_1);

% choiceprobs = NaN(length(x_grid),T);
% choiceprobs(:,T) = exp(u_1) ./ (exp(u_1) + exp(u_0));
% 
% for t = T-1:-1:1 
%     choiceprobs(:,t) = exp(u_1 + (beta .* (Fx_1 * V(:,t+1)))) ./ (exp(u_1 + (beta .* (Fx_1 * V(:,t+1)))) + exp(u_0 + (beta .* (Fx_0 * V(:,t+1)))));
% end

%%%%%%% Simulate the dataset for N buses
rng(1)

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
        if choicerand(i,j) <= choiceprobs(statematindex,j)
            choicemat(i,j) = 1;
        elseif choicerand(i,j) > choiceprobs(statematindex,j)
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
timemat = repmat(1:T,N,1);
personmat = repmat(transpose(1:N),1,T);
statedat = reshape(transpose(statemat),[],1);
choicedat = reshape(transpose(choicemat),[],1);
xnextdat = reshape(transpose(xnextmat),[],1);
persondat = reshape(transpose(personmat),[],1);
timedat = reshape(transpose(timemat),[],1);

data = [choicedat statedat xnextdat persondat timedat];
writematrix(data,'data/data_finite.csv')