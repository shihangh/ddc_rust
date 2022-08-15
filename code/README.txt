README
## GENERATE DATA ##
generate_data_inf.m
generate_data_finite.m

## PERFORM ESTIMATION ##
estimate_inf.m
estimate_finite.m

## CALCULATE LOG-LIKELIHOOD ##
rust_loglik_inf.m
rust_loglik_finite.m

## Functions used in multiple files ##
c.m : Defines a linear function used as an input to cost.m
cost.m : Defines a piecewise function that gives the utility for undertaking each option
inner_algo.m : Executes the inner-algorithm based on the contraction mapping to converge to the value functions in the infinite problem
iteration.m : Performs an iteration on the vector of Emax values (for each value of x) using the Emax function

