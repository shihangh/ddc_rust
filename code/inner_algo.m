function [Vbar_1] = inner_algo(Fx_1,Fx_0,u_0,u_1,beta,it_tol)
    Vbar_1 = log(exp(u_0) + exp(u_1)); % Start from the static case
    max_itdiff = 1;                    % Start with a value of the difference between iterations
    it_counter = 0;
        while max_itdiff > it_tol
            Vbar_0 = Vbar_1;           % Update the current Vbar with the previously computed t+1
            Vbar_1 = iteration(Vbar_0,Fx_1,Fx_0,u_0,u_1,beta); % Update the value of Vbar
            it_diff = abs(Vbar_1 - Vbar_0);
            max_itdiff = max(it_diff);
            %display(max_itdiff)
            it_counter = it_counter + 1;
            %display(it_counter)
        end
end