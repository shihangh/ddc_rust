function [negloglik,grad] = rust_loglik(data,pars,it_tol)
    % Description
    % pars - 4 times 1 vector; pars = [rc ; theta1_1; p_x0; p_x1];
    % it_tol - the tolerance for the contraction mapping
    
    % Initialise
    rc = pars(1);
    theta1_1 = pars(2);
    p_x0 = pars(3);
    p_x1 = pars(4);
    p_x2 = 1- p_x0 - p_x1;

    beta = 0.8;

    %Set up discretised state space of x
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

    % Solve the model
    Vbar = inner_algo(Fx_1,Fx_0,u_0,u_1,beta,it_tol);

    % Write the v(a,x)s
    v_0 = u_0 + (beta .* (Fx_0 * Vbar));
    v_1 = u_1 + (beta .* (Fx_1 * Vbar));

    % Compute choice probabilities
    exp_v0 = exp(v_0);
    exp_v1 = exp(v_1);
    sum_v = exp_v0 + exp_v1;
    choiceprob_1 = exp_v1 ./ sum_v;
    choiceprob_0 = exp_v0 ./ sum_v;

    % Compute negative loglik (since the optimiser in matlab is a minimiser)
    negloglik = loglik(data,choiceprob_1,choiceprob_0,Fx_0,Fx_1);
    % Log-likelihood functions:
    % Likelihood component for the choice probability
    function [logchoiceprob] = choiceprob(a,x,choiceprob_1,choiceprob_0)
        modx = floor(x./5000)+1;
        if a == 1 
            logchoiceprob = log(choiceprob_1(modx));
        elseif a == 0
            logchoiceprob = log(choiceprob_0(modx));
        else
            error('a should be 0 or 1.')
        end
    end
    % Likelihood component for the transition probability
    function [logtransprob] = transprob(x,x_f,a,Fx_0,Fx_1)
        modx = floor(x./5000)+1;
        modxf = floor(x_f./5000)+1;
        if a == 1 
            logtransprob = log(Fx_1(modx,modxf));
        elseif a == 0
            logtransprob = log(Fx_0(modx,modxf));
        else
            error('a should be 0 or 1.')
        end    
    end
    % Likelihood function for all the structural paramaters
    function [sumloglik] = loglik(data,choiceprob_1,choiceprob_0,Fx_0,Fx_1)
        N = size(data,1);
        loglikvec = zeros(N,1);
        for i = 1:N
            loglikvec(i) = choiceprob(data(i,1),data(i,2),choiceprob_1,choiceprob_0) + ...
                           transprob(data(i,2),data(i,3),data(i,1),Fx_0,Fx_1);
        end
        sumloglik = -sum(loglikvec);
    end
end