addpath 'Datasets/cifar-10-batches-mat/';

main()

function m = main()
    % load the batches
    [train_X, train_Y, train_y] = LoadBatch('data_batch_1.mat');
    [val_X, val_Y, val_y] = LoadBatch('data_batch_2.mat');
    [test_X, test_Y, test_y] = LoadBatch('test_batch.mat');
    
    K = 10; % number of lables
    N = 10000; % number of pictures 
    d = 3072; % dimentionality of each image 32x32x3 = 3072
    std_dev = 0.01;
    lambda = 0;
    
    % GDparams
    GDparams.n_batch = 100;
    GDparams.eta = 0.01;
    GDparams.n_epochs = 20;

    % initialize model W and b
    [W, b] = InitWb(K, d, K, 1, std_dev, 0);

    % size 50 and 200, lambda 0 0.1 1
    batch_sizes = [50, 200];
    lambdas = [0.0, 0.1, 1.0];
    disp('Gradient Check');
    for i = batch_sizes
        for j = lambdas
            P = EvaluateClassifier(train_X(1:i, 1), W(:, 1:i), b);
            [grad_W, grad_b] = ComputeGradients(train_X(1:i, 1), train_Y(:, 1), P, W(:, 1:i), j);
            [ngrad_b, ngrad_W] = ComputeGradsNumSlow(train_X(1:i, 1), train_Y(:, 1), W(:, 1:i), b, j, 1e-6);
            [rel_err_grad_W, rel_err_grad_b] = RelativeError(grad_W, ngrad_W, grad_b, ngrad_b);
            fprintf('batch_size: %d      lambda: %f      rel_err_grad_W: %e      rel_err_grad_b: %e \n', i, j, rel_err_grad_W, rel_err_grad_b);
        end
    end

    % lambda=0, n epochs=40, n batch=100, eta=.1
    lambda = 0;
    GDparams.n_batch = 100;
    GDparams.eta = 0.1;
    GDparams.n_epochs = 40;
    [Wstar, bstar] = MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, lambda);
    r = result(Wstar);
    acc = ComputeAccuracy(test_X, test_y, Wstar, bstar);
    fprintf('Test accuracy after eperiment 1: %f\n', acc);
    
    % lambda=0, n epochs=40, n batch=100, eta=.01
    lambda = 0;
    GDparams.n_batch = 100;
    GDparams.eta = 0.01;
    GDparams.n_epochs = 40;
    [Wstar, bstar] = MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, lambda);
    r = result(Wstar);
    acc = ComputeAccuracy(test_X, test_y, Wstar, bstar);
    fprintf('Test accuracy after eperiment 2: %f\n', acc);
    
	% lambda=.1, n epochs=40, n batch=100, eta=.01
    lambda = 0.1;
    GDparams.n_batch = 100;
    GDparams.eta = 0.01;
    GDparams.n_epochs = 40;
    [Wstar, bstar] = MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, lambda);
    r = result(Wstar);
    acc = ComputeAccuracy(test_X, test_y, Wstar, bstar);
    fprintf('Test accuracy after eperiment 3: %f\n', acc);
    
    % lambda=1, n epochs=40, n batch=100, eta=.01
    lambda = 1;
    GDparams.n_batch = 100;
    GDparams.eta = 0.01;
    GDparams.n_epochs = 40;
    [Wstar, bstar] = MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, lambda);
    r = result(Wstar);
    acc = ComputeAccuracy(test_X, test_y, Wstar, bstar);
    fprintf('Test accuracy after eperiment 4: %f\n', acc);
    
end

% load the batch
% return X: contains  the  image  pixel  data
%        Y: contains the one-hot representation of the label for each image
%        y: contains the label for each image
function [X, Y, y] = LoadBatch(filename)
    K = 10;
    N = 10000;
    batch = load(filename);
    X = double(batch.data') / 255.0;
    y = double(batch.labels') + 1.0;
    Y = zeros(K, N);
    for i = 1 : N
        Y(y(i), i) = 1;
    end

end

% initialize model W and b with random gaussian values
% with zero mean and standard deviation of 0.01
function [W , b] = InitWb(W_first, W_second, b_first, b_second, std_dev, mean)
    rng(400);
    W = randn(W_first, W_second) * std_dev + mean;
    b = randn(b_first, b_second) * std_dev + mean;
    
end

% evaluate the network
function P = EvaluateClassifier(X, W, b)
    % formula s = WX + b1^T
    one = ones(1, size(X, 2));
    s = W * X + b * one;
    
    % p = softmax(s), softmax function exp(s) / 1^T exp(s)
    exp_ = exp(s);
    exp_sum = sum(exp_, 1);
    under = ones(size(W, 1), 1) * exp_sum;
    
    P = exp_ ./ under;
    
end

% computes the cost function for a set of images
function J = ComputeCost(X, Y, W, b, lambda)
    one_over_data_magnitude = 1 / size(X, 2);
    P = EvaluateClassifier(X, W, b);
    Y_T_p = Y.' * P;
    l_cross = -log(Y_T_p);
    l_cross_sum = sum(diag(l_cross), 'all');
 
    W_squared_sum = sum(W .* W, 'all');

    J = (one_over_data_magnitude * l_cross_sum) + (lambda * W_squared_sum);
    
end

% computes the accuracy of the network%s
% predictions on a set of data
function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    
    [~, arg_max] = max(P);
    
    correct = 0;
    for i=1:size(X, 2)
        if arg_max(i) == y(i)
            correct = correct + 1;
        end
    end
    
    acc = correct / size(X, 2) * 100;

end

% Write the function that evaluates, for a mini-batch, the gradients of
% the cost function w.r.t. W and b
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    G_batch = -(Y - P);
    L_w_r_t_W = (1 / size(X, 2)) * G_batch * X.';
    L_w_r_t_b = (1 / size(X, 2)) * G_batch * ones(size(X, 2), 1);
    
    grad_W = L_w_r_t_W + (2 * lambda * W);
    grad_b = L_w_r_t_b;
    
end

% function given for assignment comparison
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);
    
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);
    
    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end
    
    for i=1:numel(W)
        
        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
        
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W(i) = (c2-c1) / (2*h);
    end
end

% mini-batch gradient descent algorithm
% to learn the network?s parameters where the updates are defined in
% equations (8, 9)
% parameters controlling thelearning algorithm: 
% n_batch the size of the mini-batches
% eta the learning rate
% n_epochs the number of runs through the whole training set
function [Wstar, bstar] = MiniBatchGD(X, Y, Xval, Yval, GDparams, W, b, lambda)
    N = size(X, 2);
    [y, ~, ~] = find(Y);
    y = y';
    [yval, ~, ~] = find(Yval);
    yval = yval';
    
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    
    costs = zeros(n_epochs, 2);
    accuracies = zeros(n_epochs, 2);
    
    for i=1:n_epochs
        for j=1:N / n_batch
            j_start = (j-1) * n_batch + 1;
            j_end = j * n_batch;
            inds = j_start:j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);

            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);

            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end
        
        costs(i, 1) = ComputeCost(X, Y, W, b, lambda);
        accuracies(i, 1) = ComputeAccuracy(X, y, W, b);
        
        costs(i, 2) = ComputeCost(Xval, Yval, W, b, lambda);
        accuracies(i, 2) = ComputeAccuracy(Xval, yval, W, b);
        
    end
    
    figure();
    x = 1 : n_epochs;
    plot(x, costs(:,1),x, costs(:,2));
    figure();
    plot(x, accuracies(:,1),x, accuracies(:,2));
    
    Wstar = W;
    bstar = b;
    
end

% computes relative error and returns for grad_W and grad_b
function [rel_err_grad_W, rel_err_grad_b] = RelativeError(grad_W, ngrad_W, grad_b, ngrad_b)
    top_W = abs(ngrad_W - grad_W);
    bottom_W = max(0.0001, abs(ngrad_W) + abs(grad_W));
    rel_err_grad_W = max(max(top_W ./ bottom_W));
    
    top_b = abs(ngrad_b - grad_b);
    bottom_b = max(0.0001, abs(ngrad_b) + abs(grad_b));
    rel_err_grad_b = max(top_b ./ bottom_b);
    
end

% visualization for the W matrix
function r = result(W)
    figure();
    
    for i=1:size(W,1)
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:))- min(im(:)));
        s_im{i} = permute(s_im(:,:,:,i), [2, 1, 3]);
    end
    
    montage(s_im, 'Size', [1,10]);
    
    r = s_im;
end