function gradient = MSE_grad(X,T,W, g)
    gradient = zeros(size(W));
    for k = 1:size(X,2)
        xk = X(:,k);
        tk = T(:,k);
        gk = g(xk,W);
        gradient = gradient + ((gk-tk) .* gk .* (1-gk))*xk.';    % elementwise multiplication    
    end
end