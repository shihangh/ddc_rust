function [negloglik,grad] = rust_loglik_finite(data,pars)
    % Description
    % pars - 4 times 1 vector; pars = [rc ; theta1_1; p_x0; p_x1];
    
    % Initialise
    rc = pars(1);
    theta1_1 = pars(2);
    p_x0 = pars(3);
    p_x1 = pars(4);
    p_x2 = 1- p_x0 - p_x1;

    beta = 0.8;
    T = max(data(:,5));
    N = max(data(:,4));

        %Set up discretised state space of x (mileage)
    x_grid = transpose(0:5000:350000); 

    % Vector of costs in each period, for each choice
    u_1 = arrayfun(@(x) cost(1,x,pars),x_grid); % Compute u(a,x) for a = 1 (71 times 1)
    u_0 = arrayfun(@(x) cost(0,x,pars),x_grid); % Compute u(a,x) for a = 0 (71 times 1)

    % Create transition matrix Fx_0 (71 times 71)
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

    % Backwards induction
    V = NaN(length(x_grid),T);
    V(:,T) = log(exp(u_1) + exp(u_0));

    for t = T-1:-1:1 
       V(:,t) = log(exp(u_1 + (beta .* (Fx_1 * V(:,t+1)))) + exp(u_0 + (beta .* (Fx_0 * V(:,t+1)))));
    end

    V_0 = NaN(length(x_grid),T);
    V_1 = NaN(length(x_grid),T);
    V_0(:,T) = exp(u_0);
    V_1(:,T) = exp(u_1);
    for t = T-1:-1:1
       V_0(:,t) = exp(u_0 + (beta .* (Fx_0 * V(:,t+1))));
       V_1(:,t) = exp(u_1 + (beta .* (Fx_1 * V(:,t+1))));
    end

    choiceprobs = V_1 ./ (V_0 + V_1);

    % Compute negative loglik (since the optimiser in matlab is a minimiser)
    loglik_choice_it = NaN(max(data(:,4)),max(data(:,5)));
    loglik_trans_it = NaN(max(data(:,4)),max(data(:,5)));

    for i = 1:N
        for j = 1:T
            a = data(data(:,4) == i & data(:,5) == j,1);
            xt = data(data(:,4) == i & data(:,5) == j,2);
            xt1 = data(data(:,4) == i & data(:,5) == j,3);
            modx = floor(xt ./ 5000)+1;
            modxt = floor(xt1 ./5000)+1;        
            if a == 1 
                loglik_choice_it(i,j) = log(choiceprobs(modx,j));
                loglik_trans_it(i,j) = log(Fx_1(modx,modxt));
            elseif a == 0
                loglik_choice_it(i,j) = log(1-choiceprobs(modx,j));
                loglik_trans_it(i,j) = log(Fx_0(modx,modxt));            
            else
                error('a should be 0 or 1.')
            end        
        end
    end

    loglik_i = sum(loglik_choice_it + loglik_trans_it,2);
    totloglik = sum(loglik_i,1);
    negloglik = -totloglik;

end