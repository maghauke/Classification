function [W,m] = gradient_descent(gradient,W0,alpha)
    % alpha, learning rate
    % gradient, output from MSE_grad
    % W0, initial weighting matrix
    W = W0;
    iterate = true;
    m = 1;
    N = 2*1e4;      % number of iterations
    tol = 1e-3;     % 
    while iterate && m < N
       mse_grad = gradient(W);
       W = W - alpha * mse_grad;
       iterate = norm(mse_grad,inf) > tol;
       m = m + 1;
    end
end