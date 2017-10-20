%Logistic Regression

x = load('logistic_x.txt'); %training data
y = load('logistic_y.txt'); %output values

x = [ones(size(x, 1), 1) x]; %intercept

nsamples = size(x,1);
nparams = size(x,2);


theta = zeros(nparams, 1); %first initialize with all zeros

n_iterations = 30;

for i = 1:n_iterations
    ytx = y .* (x * theta);
    hyx = 1 ./ (1 + exp(ytx));
    grad = (-1/nsamples) * (x' * (hyx .* y)); %gradient of loss function
    grad2 = (1/nsamples) * (x' * (diag(hyx .* (1 - hyx)) * x)); %diag is clean way to get hessian
    theta = theta - grad2 \ grad; %backslash is matrix division
end

figure;
hold on;
plot(x(y > 0, 2), x(y > 0, 3), 'bo');
plot(x(y < 0, 2), x(y < 0, 3), 'rx');
x1 = min(x(:,2)):0.1:max(x(:,2));
x2 = -(theta(1) / theta(3)) - (theta(2) / theta(3)) * x1; %when model is 0.5, then theta^T*x = 0. rearrange terms and we get this
plot(x1, x2);
xlabel("x1");
ylabel("x2");
title("Binary Classification w/ Logistic Regression")


    
