function [Vit] = iteration(Vbar,Fx_1,Fx_0,u_0,u_1,beta)
    it_0 = exp(u_0 + (beta .* (Fx_0 * Vbar)));  %VBar update given no replacement of the bus engine
    it_1 = exp(u_1 + (beta .* (Fx_1 * Vbar)));  %VBar update given replacement of the bus engine
    it = it_0 + it_1; 
    Vit = log(it);
end
