function [W,n] = gradient_descent(gradient,W0,alpha) 
    W = W0;
    iterate = true;
    n = 1;
    N = 1000;
    tol = 1e-6;
    while iterate && n < N
       mse_grad = gradient(W);
       W = W - alpha * mse_grad;
       iterate = norm(mse_grad) > tol;
       n = n+1;
    end
end