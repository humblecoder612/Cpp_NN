%% MATLAB Script
% DATA
clc
clear
close all

load('ex3data1.mat')
load('ex3weights.mat')

m = 100;
m_T = 5000;
theta = cell(2,1);

%theta1 = rand(25,401);
%theta2 = rand(10,26)

theta{1} = Theta1;
theta{2} = Theta2;

X = X';
result = zeros(10 , m_T);
for z = 1:m_T
    result(y(z) , z) = 1;
end

%tester = forwardpropagation(X(: , 2) , theta , [1 1]);
%tester{3}

lambda = 0;
alpha = 0.01;
max_iter = 800;

for i= 1:m_T
test = X(: , i);
avg = sum(test)/400;
st = std(test);

X(:,i) = (test - avg*ones(size(test))) / st; 
end

for g = 1:2500
%[i j] = rand( 1, 2);
i = ceil( 5000 * rand(1,2));
j = i(2);
i = i(1);
temp = X(: , i);
X(: , i) = X(: , j);
X(: , j) = temp;

temp = result(: , i);
result(: , i) = result(: , j);
result(: , j) = temp;
end

fprintf("Making Model ...")

j = 0;
for g = 1:50
j = j + 100;

    BatchX = X(: ,  (j - 100 + 1):j );
    BatchR = result(: , (j - 100 + 1):j );

[ theta , C ]  = learn(BatchX , theta , BatchR , m , lambda , alpha , max_iter);

clc
fprintf("Percentage Done : %2f " , 2*g )

end

plot( linspace(1 , max_iter , max_iter ) , C)

tester = forwardpropagation(X(: , 2) , theta , bias(theta));
%%
% Functions
function [t , cost_saver ] = learn(X , theta , result , m , lambda , alpha , max_iter)
    
    L = size(theta , 1) +1;
    cost_saver = ones(max_iter , 1);
    for g = 1:max_iter
        J_p = J_prime(X , theta , result , m , lambda);
        for j = 1:L-1
            theta{j} = theta{j} - alpha*J_p{j};
        end
        cost_saver(g) = cost(X , theta , result , m);
    end
    
    t = theta;
end

function J_p = J_prime( X , theta , result , m , lambda )
    B = bias(theta);
    bigdel = init_del(theta);
    L = size(theta , 1) + 1;
    
    for p = 1:m
        net = forwardpropagation(X( : , p) , theta , B);
        del = error(net , theta , result(: , p));
        
        for j = 1:L-1
            bigdel{j} = bigdel{j} + del{j+1}*(net{j})'; 
        end
        
    end
    
    J_p = cell(size(theta));
    
    for j = 1:L-1
        J_p{j} = (bigdel{j} + lambda*theta{j})/m; 
    end
    
end

function del = error( net , theta , y)
% net is along with the biases

L = size(net , 1);

del = cell( L , 1);
del{L} = (net{L} - y) ;

for j=1:(L-1)
    p = L-j;
    if (p+1 ~= L)
        del{p+1} = remove_bias(del{p+1});
    end
    del{p} = (theta{p}')*del{p+1}.*net{p}.*(ones(size(net{p})) - net{p});
end

end

function net = forwardpropagation(x , theta , B)
% x is given without a bias , bias vector B is provided independantly with
% length L-1

L = size(theta,1) + 1;
net = cell( L , 1);

net{1} = x; 
for i = 1:L-1
    net{i} = [B(i) ; net{i}];
    net{i+1} = sigmoid(theta{i} * net{i});
    %net{i}
end

end

function bigdel = init_del(theta)

L = size(theta , 1) + 1;
bigdel = theta;

for i=1:(L-1)
    bigdel{i} = zeros(size(bigdel{i}));
end

end

function B = bias(theta)
    L = size(theta , 1);
    B = ones(1 , L );
end

function b = remove_bias( v )
    
    h = size(v , 1);
    b = zeros( h - 1 , 1);
    for i=2:h
        b(i - 1) = v(i);
    end
end

function J = cost( X , theta , result , m)
    J = 0;
    for i = 1:m
        net = forwardpropagation(X(: , i) , theta ,result(: , i));
        J = J + sum((net{size(theta , 1) + 1} - result(: , i)).^2);
    end
    J = J/m;
end
