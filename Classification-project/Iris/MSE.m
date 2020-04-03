function mse = MSE(X, T, W, g, N)
    mse = 0;
    for k = 1:N
        gk = g(X(:,k),W);
        tk = T(:,k);        
        mse = mse + 0.5 * (gk-tk)'*(gk-tk);
    end
end

