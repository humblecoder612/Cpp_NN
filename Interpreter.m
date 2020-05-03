imdata = imread("seven2.jpg");
imshow(imdata)
imdata = rgb2gray(imdata);
imshow(imdata)
imdata = imresize(imdata , [20 20]);

test = zeros(400 , 1);

for i = 1:400
    if mod(i,20) == 0
        k = 20;
    else
        k = mod(i,20);
    end
    test(i) = imdata(  k  , (i - k)/20 + 1  );

end

%test = X(:,4968);
%result(: , 4968)

avg = sum(test)/400;
st = std(test);

test = (test - avg*ones(size(test))) / st; 

fin = forwardpropagation(test , theta , [1 1]);

    find ( fin{3} == max(fin{3}))
    
    sprintf( "with probability %f " , 100*max(fin{3}))


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