% testing projection of matrix W on a set of symmetric centered matrices

n=30;
W = randn(n);

%closed form solution
Y = 0.5*(W+W');
Y = dc(Y);


%compare with solution of quadprog
f_handle = @(y)(sum((y-W(:)).^2));

J=(1:n^2).';
I=reshape(J,n,n).';
T=sparse(I,J,1,n^2,n^2); %transposition operator
Aeq=[speye(n^2)-T;...         %symmetry constraint
    kron(ones(1,n),speye(n))]  ;      %other constrain
c = zeros(n,1);    
beq=[zeros(n^2,1); c(:)];
y=fmincon(f_handle,W(:), [],[],Aeq,beq);

Y_=reshape(y,n,n);
disp(abs(norm(Y_-Y)))







function [CX] = dc(X,n)
% double centering
if nargin <2
    n = size(X,1);
end
rowSum = sum(X,1);
colSum = sum(X,2);
allSum = sum(colSum);
CX = X - (1/n)*(repmat(rowSum,n,1)+repmat(colSum,1,n))+(1/n.^2)*allSum;

end