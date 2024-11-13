function [U,S,V] = dualsvd(A,k)

% dualsvd: computes the rank-k approximation of a dual complex matrix 
%          under the quasi-metric d_*

% Input:
%   A = A(:,:,1) + A(:,:,2)*epsilon
%   k: rank-k approximation

% Output:
%   U = U(:,:,1) + U(:,:,2)*epsilon
%   S = S(:,:,1) + S(:,:,2)*epsilon
%   V = V(:,:,1) + V(:,:,2)*epsilon

% Optimization problem:
%                       min  d_*^2(A - U*S*V'),
%                       s.t. U'*U=I, V'*V=I, S is diagonal positive.

% By Weiyang Ding @Fudan July 16, 2022

% For more details, please refer to 
%
% Tong Wei, Weiyang Ding, and Yimin Wei, Singular Value Decomposition of 
% Dual Matrices and Its Application to Traveling Wave Identification in 
% the Brain, SIAM J. Matrix Anal. Appl., 45 (2024), pp. 634–660.
%
% https://doi.org/10.1137/23M1556642

[Ut,St,Vt] = svd(A(:,:,1),'econ');
s = diag(St);
if nargin == 1
    ind = s>0;
else
    ind = 1:k;
end
U(:,:,1) = Ut(:,ind);
S(:,:,1) = St(ind,ind);
V(:,:,1) = Vt(:,ind);
s = s(ind);
s = s.*s;
D = 1./(s'-s);
D(isinf(D)) = 0;
R = U(:,:,1)'*A(:,:,2)*V(:,:,1);
P = (R*S(:,:,1)+S(:,:,1)*R').*D;
Q = (S(:,:,1)*R+R'*S(:,:,1)).*D;
s = diag(S(:,:,1));
S(:,:,2) = diag(diag(R) - diag(P).*s - s.*conj(diag(Q)));
U(:,:,2) = U(:,:,1)*(P-R/S(:,:,1)) + A(:,:,2)*V(:,:,1)/S(:,:,1);
V(:,:,2) = V(:,:,1)*(Q-R'/S(:,:,1)) + A(:,:,2)'*U(:,:,1)/S(:,:,1);

end