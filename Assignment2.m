addpath Datasets/cifar-10-batches-mat/;
% main()
% CoarseToFine(-5, -1, 5000, 20)
% CoarseToFine(-3, -1, 5000, 30)
% FinalTest(1.733825e-03, 9000)

% final test results
function final = FinalTest(lambda, val_size)
    rng(400);
    d = 3072;
    m = 50;
    
    [X1, Y1,  y1] = LoadBatch('data_batch_1.mat');
    [X2, Y2,  y2] = LoadBatch('data_batch_2.mat');
    [X3, Y3,  y3] = LoadBatch('data_batch_3.mat');
    [X4, Y4,  y4] = LoadBatch('data_batch_4.mat');
    [X5, Y5,  y5] = LoadBatch('data_batch_5.mat');
    
    trainX = [X1, X2, X3, X4, X5(:, 1:val_size)];
    valX = X5(:, val_size + 1:end);
    
    trainY = [Y1, Y2, Y3, Y4, Y5(:, 1:val_size)];
    valY = Y5(:, val_size + 1:end);

    trainy = [y1; y2; y3; y4; y5(1:val_size)];
    valy = y5(val_size + 1:end);
    
    [testX, testY,  testy] = LoadBatch('test_batch.mat');
    
    K = size(trainY, 1);
    
    [W, b] = InitWb(K, d, m);
    
    n_step = 1250;
    batch_size = 100;
    n_cycles = 3;
    GDparams.n_step = n_step;
    GDparams.n_batch = batch_size;
    GDparams.n_cycles = n_cycles;
    
    [Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);
    acc = ComputeAccuracy(testX, testY, Wstar, bstar);
    fprintf('\nFinal test accuracy with best lambda: %f', acc);

end

% find the best lambda
function lam = CoarseToFine(l_min, l_max, val_size, lam_itr)
    rng(400);
    d = 3072;
    m = 50;
    
    [X1, Y1,  y1] = LoadBatch('data_batch_1.mat');
    [X2, Y2,  y2] = LoadBatch('data_batch_2.mat');
    [X3, Y3,  y3] = LoadBatch('data_batch_3.mat');
    [X4, Y4,  y4] = LoadBatch('data_batch_4.mat');
    [X5, Y5,  y5] = LoadBatch('data_batch_5.mat');
    
    trainX = [X1, X2, X3, X4, X5(:, 1:val_size)];
    valX = X5(:, val_size + 1:end);
    
    trainY = [Y1, Y2, Y3, Y4, Y5(:, 1:val_size)];
    valY = Y5(:, val_size + 1:end);

    trainy = [y1; y2; y3; y4; y5(1:val_size)];
    valy = y5(val_size + 1:end);
    
    [testX, testY,  testy] = LoadBatch('test_batch.mat');
    
    K = size(trainY, 1);
    
    [W, b] = InitWb(K, d, m);
    
    n_s = 900;
    batch_size = 100;
    cycles = 3;
    GDparams.n_step = n_s;
    GDparams.n_batch = batch_size;
    GDparams.n_cycles = cycles;
    
    acc = zeros(lam_itr, 1);
    lambdas = zeros(lam_itr, 1);
    for i=1:lam_itr
        l = l_min + (l_max - l_min)*rand(1, 1);
        lambda = 10^l;
        lambdas(i) = lambda;

        [Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);
        acc(i) = ComputeAccuracy(testX, testY, Wstar, bstar);
        disp(i);
    end
    
    
    [max_acc, ids] = maxk(acc, 3);
    
    fprintf('\n\nMax lambda1: %e, max acc2: %e', lambdas(ids(1)), max_acc(1));
    fprintf('\n\nMax lambda2: %e, max acc2: %e', lambdas(ids(2)), max_acc(2));
    fprintf('\n\nMax lambda2: %e, max acc2: %e', lambdas(ids(3)), max_acc(3));
    
    fig = figure;
    scatter(lambdas, acc);
    title('lambda plot');
    xlabel('lambda')
    ylabel('acc')
    
end

% test gradient, figure 3 and figure 4
function m = main()
        % load the batches
    [train_X, train_Y, train_y] = LoadBatch('data_batch_1.mat');
    [val_X, val_Y, val_y] = LoadBatch('data_batch_2.mat');
    [test_X, test_Y, test_y] = LoadBatch('test_batch.mat');
    
    K = 10;
    d = 3072;
    m = 50;

    % initialize model W and b
    [W, b] = InitWb(K, d, m);

    % size 50 and 200, lambda 0 0.1 1
%     batch_sizes = [200, 50];
%     lambdas = [0.0, 0.1, 1.0];
%     disp('Gradient Check');
%     for i = batch_sizes
%         W{1} = W{1}(:, 1:i);
%         W{2} = W{2}(:, :);
%         for j = lambdas
%             [grad_W, grad_b] = ComputeGradients(train_X(1:i, 1:2), train_Y(:, 1:2), W, b, j);
%             [ngrad_b, ngrad_W] = ComputeGradsNumSlow(train_X(1:i, 1:2), train_Y(:, 1:2), W, b, j, 1e-5);
%             [rel_err_grad_W, rel_err_grad_b] = RelativeError(grad_W, ngrad_W, grad_b, ngrad_b);
%             fprintf('batch_size: %d      lambda: %f      \nrel_err_grad_W_1: %e     rel_err_grad_W_2: %e      \nrel_err_grad_b_1: %e      rel_err_grad_b_2: %e \n\n', i, j, rel_err_grad_W{1}, rel_err_grad_W{2}, rel_err_grad_b{1}, rel_err_grad_b{2});
%         end
%     end
    
   
    % replicate figure 3
    lambda = 0.01;
    GDparams.n_batch = 100;
    GDparams.n_step = 500;
    GDparams.n_cycles = 1;
    [Wstar, bstar] = MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, lambda);
%     r = result(Wstar);
    acc = ComputeAccuracy(test_X, test_Y, Wstar, bstar);
	fprintf('Test accuracy after experiment 1: %f\n', acc);

    % replicate figure 4
    lambda = 0.01;
    GDparams.n_batch = 100;
    GDparams.n_step = 800;
    GDparams.n_cycles = 3;
    [Wstar, bstar] = MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, lambda);
%     r = result(Wstar);
    acc = ComputeAccuracy(test_X, test_Y, Wstar, bstar);
	fprintf('Test accuracy after experiment 2: %f\n', acc);
end

% load the batch
% return X: contains  the  image  pixel  data
%        Y: contains the one-hot representation of the label for each image
%        y: contains the label for each image
function [X, Y, y] = LoadBatch(filename)
    K = 10;
    N = 10000;
    batch = load(filename);
    X = double(batch.data');
    y = double(batch.labels) + 1.0;
    Y = zeros(K, N);
    for i = 1 : N
        Y(y(i), i) = 1;
    end
    
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
    
end

% initialize model W and b with random gaussian values
function [W , b] = InitWb(K, d, m)
    rng(400);
    
    W_1 = randn(m,d) * 1.0/sqrt(d);
    W_2 = randn(K,m) * 1.0/sqrt(m);
    b_1 = zeros(m,1);
    b_2 = zeros(K,1);
    
    W = {W_1, W_2};
    b = {b_1, b_2};
    
end

% evaluate the network
function [P, H] = EvaluateClassifier(X, W, b)
    % formula s = WX + b1^T
    one = ones(1, size(X, 2));
    s_1 = W{1} * X + b{1} * one;
    
    H = max(0, s_1);
    s_2 = W{2} * H + b{2} * one;
    
    % p = softmax(s), softmax function exp(s) / 1^T exp(s)
    exp_ = exp(s_2);
    exp_sum = sum(exp_, 1);
    under = ones(size(W, 1), 1) * exp_sum;
    
    P = exp_ ./ under;
end

% computes the cost function for a set of images
function [J, loss] = ComputeCost(X, Y, W, b, lambda)  
    p = EvaluateClassifier(X, W, b);
    loss_func = -log(sum(Y .* p, 1));
    
    W1 = W{1};
    W2 = W{2};
    reg_sum = sum(sum(W1 .* W1, 'double'), 'double');
    reg_sum = reg_sum + sum(sum(W2 .* W2, 'double'), 'double');
   
    reg_term = lambda * reg_sum;
    loss = mean(loss_func);
    J = loss + reg_term;
end

% computes the accuracy of the network%s
% predictions on a set of data
function acc = ComputeAccuracy(X, y, W, b)
    [p, ~] = EvaluateClassifier(X, W, b);
    num_correct = 0;
    num_img = size(X, 2);
    [~, argmax] = max(p);
    for img = 1:num_img
        [~, argmaxy] = max(y(:,img));
        if argmax(img) == argmaxy
            num_correct = num_correct + 1;
        end         
    end
    acc = double(num_correct)/num_img * 100;
end

% Write the function that evaluates, for a mini-batch, the gradients of
% the cost function w.r.t. W and b
function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
    [P, H] = EvaluateClassifier(X, W, b);
    G_batch = -(Y - P);
    
    L_w_r_t_W_2 = (1 / size(X, 2)) * G_batch * H.';
    grad_W_2 = L_w_r_t_W_2 + (2 * lambda * W{2});
    
    L_w_r_t_b_2 = (1 / size(X, 2)) * G_batch * ones(size(X, 2), 1);
    grad_b_2 = L_w_r_t_b_2;
    
    G_batch = (W{2}).' * G_batch;
    G_batch = G_batch .* (H > 0);

    L_w_r_t_W_1 = (1 / size(X, 2)) * G_batch * X.';
    grad_W_1 = L_w_r_t_W_1 + (2 * lambda * W{1});
    
    L_w_r_t_b_1 = (1 / size(X, 2)) * G_batch * ones(size(X, 2), 1);
    grad_b_1 = L_w_r_t_b_1;
    
    grad_W = {grad_W_1, grad_W_2};
    grad_b = {grad_b_1, grad_b_2};
    
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));

        for i=1:length(b{j})

            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            [c1, ~] = ComputeCost(X, Y, W, b_try, lambda);

            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            [c2, ~] = ComputeCost(X, Y, W, b_try, lambda);

            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));

        for i=1:numel(W{j})

            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            [c1, ~] = ComputeCost(X, Y, W_try, b, lambda);

            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);

            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end

% network with cyclical learning rates
function [Wstar, bstar] = MiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, W, b, lambda)
    n_batches = GDparams.n_batch;
    n_s = GDparams.n_step;
    cycles = GDparams.n_cycles;
    
    plot_idx = 1;
    
    eta_min = 1e-5;
    eta_max = 1e-1;
    step = (eta_max - eta_min)/n_s;
    itr = 2*cycles*n_s;
    eta = eta_min;
    
    N = size(X_train, 2);
    
    update = step; 
    batch_start = 1;
        
    % cost, loss, accuracy, and eta matrices
    costs = zeros(itr / 100, 2);
    accs = zeros(itr / 100, 2);
    losses = zeros(itr / 100, 2);
    etas = zeros(itr / 100, 1);
    
    for t=1:itr
        
        if batch_start >= N
            batch_start = 1;
        end

        idx = batch_start : min(batch_start + n_batches -1, N);
        
        Xbatch = X_train(:, idx);
        Ybatch =  Y_train(:, idx);
        
        %update starting index
        batch_start = batch_start + n_batches;

        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
             
        % update the params
        W{1} = W{1} - eta * grad_W{1};
        W{2} = W{2} - eta * grad_W{2};
        b{1} = b{1} - eta * grad_b{1};
        b{2} = b{2} - eta * grad_b{2};
        
        % save the cost, loss and accuracy after each update
        if mod(t,100)==0
            [c_train, l_train] = ComputeCost(X_train, Y_train, W, b, lambda);
            [c_val, l_val] = ComputeCost(XVal, YVal, W, b, lambda);
            
            costs(plot_idx, 1) = c_train;
            costs(plot_idx, 2) = c_val;
            
            losses(plot_idx, 1) = l_train;
            losses(plot_idx, 2) = l_val;
          
            accs(plot_idx, 1) = ComputeAccuracy(X_train, Y_train, W, b);
            accs(plot_idx, 2) = ComputeAccuracy(XVal, YVal, W, b);
            
            etas(plot_idx) = eta;
            
            plot_idx = plot_idx + 1;
        end

        % update eta
        eta = eta + update;
        if eta >= eta_max
            eta = eta_max;
            update = -step;
        elseif eta <= eta_min
            eta = eta_min;
            update = step;
        end
    end
    
    % plotting costs, losses, accuracies, and etas
    x = 1 : plot_idx - 1;
    plot(x*100, costs(:, 1), x*100, costs(:, 2));
    title('Cost plot');
    xlabel('update step')
    ylabel('cost')
    figure();
    
    plot(100*x, losses(:, 1), 100*x, losses(:, 2));
    title('Loss plot');
    xlabel('update step')
    ylabel('loss')
    figure();
    
    plot(100*x, accs(:, 1), 100*x, accs(:, 2));
    title('Accuracy plot');
    xlabel('update step')
    ylabel('accuracy')
    
%     figure();
%     plot(x, etas);
%     title('Eta plot');
%     xlabel('update step')
%     ylabel('eta')

    Wstar = W;
    bstar = b;
end

% Relative error function
function [grad_W_err, grad_b_err] = RelativeError(grad_W, ngrad_W, grad_b, ngrad_b)  
    numerator_W = abs(ngrad_W{1} - grad_W{1});
    denominator_W = max(0.0001, abs(ngrad_W{1}) + abs(grad_W{1}));
    grad_W1_err = max(max(numerator_W ./ denominator_W));
    
    numerator_W = abs(ngrad_W{2} - grad_W{2});
    denominator_W = max(0.0001, abs(ngrad_W{2}) + abs(grad_W{2}));
    grad_W2_err = max(max(numerator_W ./ denominator_W));
    
    numerator_b = abs(ngrad_b{1} - grad_b{1});
    denominator_b = max(0.0001, abs(ngrad_b{1}) + abs(grad_b{1}));
    grad_b1_err = max(numerator_b ./ denominator_b);
    
    numerator_b = abs(ngrad_b{2} - grad_b{2});
    denominator_b = max(0.0001, abs(ngrad_b{2}) + abs(grad_b{2}));
    grad_b2_err = max(numerator_b ./ denominator_b);
    
    grad_W_err = {grad_W1_err, grad_W2_err};
    grad_b_err = {grad_b1_err, grad_b2_err};
    
end