function Dnorm = DKFnorm(A,p,k)

% DKFnorm: calculates the dual-valued Ky Fan p-k-norm of a dual matrix A

% Input:
%   A = A(:,:,1) + A(:,:,2)*epsilon
%   p: 1 <= p < infty
%   k: 1 <= k <= size(A,2)

% Output:
%   Dnorm = Dnorm(:,1) + Dnorm(:,2)*epsilon

As = A(:,:,1);
Ai = A(:,:,2);
[U,S,V] = svd(As);

% consider the case that t=1 and s=0, that is, 
% the multiplicity of sigma_k is 1

U1 = U(:,1:k-1);
U2 = U(:,k);

V1 = V(:,1:k-1);
V2 = V(:,k);
S1 = S(1:k-1,1:k-1);

lam = U2'*Ai*V2;

sigmak = diag(S(1:k,1:k));
Dnorm(:,1) = (sum(sigmak.^p))^(1/p);
Dnorm(:,2) = (1/(Dnorm(:,1))^(p-1)) * (trace(Ai'*U1*(diag((diag(S1)).^(p-1)))*V1')...
    + S(k,k)^(p-1)*lam); 

end