function [z, x, pi, indices, exitflag] = RSM(A, b, c, m, n)
% Solves min cx s.t. Ax=b, x>=0
% exitflag is 1 if solved successfully and -1 if unbounded
% Performs a Phase I procedure starting with an all artificial basis
% and then calls function simplex

% initialising p1 costs, inverse basis matrix and augmented Amatrix, and p2 costs
c1 = [zeros(n,1);ones(m,1)];
c2 = [c;zeros(m,1)];
IBmatrix = eye(m);
indices = [n+1:1:n+m]';
A2 = [A,eye(m)];

% conducting phase 1 simplex method
[z, x,pi, indices, ~] = simplex(A,b,c1,m,n,IBmatrix, indices, 1);
% if cost isn't 0 then problem is infeasible
if (z ~= 0)
    exitflag = 1;
    return;
end

% conducting phase 2 simplex method
[z, x, pi, indices, exitflag] = simplex(A, b, c2, m, n, inv(A2(:,indices)), indices, 2);

function [z, x, pi, indices, exitflag] = simplex(A, b, c, m, n, IBmatrix, indices,phase)
% Solves min cx s.t. Ax=b, x>=0
% starting with basic variables listed in vector indices
% and basis matrix Bmatrix
% exitflag is 1 if solved successfully and -1 if unbounded
% returns optimal vector x and its value z, along with pi, and indices of basic variables

% initialising basic cost and finding xbasic
unsolved = true;
x1 = zeros(m+n,1);
cb = c(indices,1);
xb = IBmatrix*b;

while unsolved
    % calculating pi vector
    pi = (cb'*IBmatrix)';
    % finding entering variable, basic costs only so artificial can't enter
    [as,cs,s] = findenter(A,pi,c(1:n,1),indices,n);
    % if no entering variable is found then we are optimal
    if s == 0
        % updating solution vector and calculating optimal cost
        x1(indices) = xb;
        x = x1(1:n,1);
        z = c'*x1;
        exitflag = 0;       % simplex has reached optimality
        break
    end
    
    % finding the leaving variable
    leave = findleave(IBmatrix, as, xb, phase, indices,n);
    
    % if no leaving variable can be found LP is unbounded
    if leave == 0
        % updating solution vector and calculating current cost
        x1(indices) = xb;
        x = x1(1:n,1);
        z = c'*x1;
        exitflag = -1;      % LP is unbounded
        break
    end
    
    % updating IBmatrix, cb and xb
    [IBmatrix, indices, cb,xb] = updateGJ(IBmatrix, indices, cb, cs, as, s, leave,xb);
end

function [as, cs, s] = findenter(Amatrix, pi, c, indices, n)
% Given the complete m by n matrix Amatrix,
% the complete cost vector c with n components
% the vector pi with m components
% findenter finds the index of the entering variable and its column
% It returns the column as, its cost coefficient cs, and its column index s
% Returns s=0 if no entering variable can be found (i.e. optimal)

for i=1:n
    % pricing only non-basic columns
    if(sum(indices==i)==0)
        % calculating reduced cost
        rs = c(i,1)-pi'*Amatrix(:,i);
        if(rs<-1e-6)
            s = i;
            as = Amatrix(:,i);
            cs = c(i,1);
            return;
        end
    end
    
    % if all rc >= 0 then we are optimal
    as = 0;
    cs = 0;
    s = 0;
end

function [leave] = findleave(IBmatrix, as, xb, phase, indices, n)
% Given entering column as and vector xb of basic variables
% findleave finds a leaving column of basis matrix Bmatrix
% It returns 0 if no column can be found (i.e. unbounded)

vec = IBmatrix*as;           % calculates most important vector in simplex
ratio = xb./vec;             % finds the ratio for all basic variables

% if phase 2, conduct extended leaving variable
% checking if there are artificial variables with a non-0 denominator
if((phase==2)&&(sum(((vec~=0)&(indices>n)))))
    % remove artificial variable with a non-0 denominator
    [~,leave] = max((vec~=0)&(indices>n));
    return
end
% else normal leaving variable
if (sum(vec(vec>0))==0)      % if no denominator is positive then unbounded
    leave = 0;
else
    % else normal leaving variable
    ratio(ratio<0)=NaN;        % negative ratios can't be chosen
    ratio(vec<=0)=NaN;         % non-positive denominators can't be chosen
    [~,leave] = min(ratio);
end

function [IBmatrix, indices, cb,xb] = updateGJ(IBmatrix, indices, cb, cs, as, s, leave,xb)
% Bmatrix is current m by m basis matrix
% indices is a column vector current identifiers for basic variables in
% order of B columns
% cb is a column vector of basic costs in the order of B columns
% as is the entering column
% s is the index of the entering variable
% leave is the column (p) of the basis matrix that must leave
% (not its variable index t)
% update replaces column leave of Bmatrix with as to give newBmatrix
% replaces row leave of indices with enter to give newindices
% replaces row leave of cb with cs to give newcb


GJ = [xb,IBmatrix,IBmatrix*as];     % Forming augmented matrix
ratio = GJ(:,end)./GJ(leave,end);   % Finding row ratios for GJ elimination
row = GJ(leave,:);                  % Saving pivot row

% conducting GJ elimination
for i = 1:length(ratio)
    if i ~= leave
        GJ(i,:) = GJ(i,:)- ratio(i,1).*row;     % eliminating non-pivot row
    else
        GJ(i,:) = GJ(i,:)*(1/GJ(i,end));        % reducing pivot row to 1
    end
end

% returning new xb, IBmatrix, indices and cb
xb = GJ(:,1);
IBmatrix = GJ(:,2:end-1);
indices(leave,1) = s;
cb(leave,1) = cs;
