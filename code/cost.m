function [u] = cost(a,x,pars)
    rc = pars(1);

    if a == 1
       u = -(c(0,pars) +  rc);
    elseif a == 0
        u = -c(x,pars);
    else
        error('a should be 0 or 1.')
    end
end