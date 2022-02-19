%% Accuracy of correspondences

% Accuracy metric used here gives very low numbers for all the automatic
% methods. I think the problem is because we have very few points so 
% nested CV is a bit of a stretch (unable to find optimal parameters well)

cd('D:\MATLAB2021b\R2021b\bin\97113\Coursework1');
% Loading data
clear all; close all; clc;


%% Load Data
% HG images
I1 = 7; % 1  4
I2 = 8; % 3  5

%%%% Manual method to find correspondences
load(['.\HG_Matrix\manual_Pairs', num2str(I1), '_', num2str(I2), '.mat']);
% obtain inlier points
fixedPoints = pointshl;
movingPoints = pointshr;


%%%% Automatic (SURF) method to find correspondences
load(['.\HG_Matrix\Auto_HG_SURF', num2str(I1), '_', num2str(I2), '.mat']);
% % obtain points
% fixedPoints = double(matchedPr.Location(inliers, :));
% movingPoints = double(matchedPl.Location(inliers, :));

% obtain points including outliers
fixedPoints = double(matchedPr.Location);
movingPoints = double(matchedPl.Location);

% %%% Automatic (Harris) method to find correspondences
% load(['.\HG_Matrix\Auto_HG_Harris', num2str(I1), '_', num2str(I2), '.mat']);
% % obtain points
% % fixedPoints = double(matchedPr.Location(inliers, :));
% % movingPoints = double(matchedPl.Location(inliers, :));
% 
% % obtain points including outliers
% fixedPoints = double(matchedPr.Location);
% movingPoints = double(matchedPl.Location);

%%

% Nested CV
corresPoints = [fixedPoints movingPoints];

rng(223);
k = 3;
n = length(corresPoints(:,1));

if(n>11)
    if (n<30)
        disp("Warning: Very small sample size")
    end

    % Since the number of data points for correspondences for some methods is 
    % small (<30), randomly generating the indices vector is not consistent 
    % and some folds may have a small sample size. Due to this, the indices 
    % vector is continously generated until the data set is evenly split

    indices = zeros(k);
    evenlySplit = 0;
    while(~evenlySplit)
        for i = 1:k
            if~(length(indices(indices==i))<((n/k)-2) || length(indices(indices==i))>((n/k)+2))
                evenlySplit = 1;
            end
        end
        indices = crossvalind('Kfold',n,k);
    end
    split = [length(indices(indices==1)) length(indices(indices==2)) length(indices(indices==3))];

    n_accurate = 0;
    CV_error = zeros(1,k);
    all_opt_h = cell(1,k);
    accuracy = zeros(1,k);
    % Outer loop; k-fold cross validation (for accuracy)
    for i = 1:k
        outer_train = corresPoints(indices~=i,:);
        outer_test = corresPoints(indices==i,:);

        outer_train_x = outer_train(:,1:2);
        outer_train_y = outer_train(:,3:4);

        % Inner loop; leave one out cross validation (LOOCV for parameter estimation)
        inner_idx = 1:length(outer_train);

        LOOCV_error = zeros(1,length(outer_train));
        all_h = cell(1,length(outer_train));
        for j = 1:length(outer_train)
            sub_train = outer_train(inner_idx~=j,:);
            validation = outer_train(inner_idx==j,:);

            train_x = sub_train(:,1:2);
            train_y = sub_train(:,3:4);

            % Estimate homography matrix (parameters) from sub_train data
            h = homography_solver(train_x', train_y');
            h = (1/h(3,3))*h;
            all_h{j} = h;

            test_x = [validation(:,1:2) 1];
            test_y = [validation(:,3:4) 1];

            % Calculate LOOCV error (mean square error)
            predict = h*test_x';
            predict = predict/predict(end);
            L2_dist = sqrt(sum((test_y'-predict).^2));
            mse = mean(L2_dist);
            LOOCV_error(j) = mse;

        end

        % Select optimal parameters with lowest LOOCV error
        opt_h = all_h{LOOCV_error==min(LOOCV_error)};
        all_opt_h{i} = opt_h;

        outer_test_x = [outer_test(:,1:2) ones(length(outer_test(:,1)),1)];
        outer_test_y = [outer_test(:,3:4) ones(length(outer_test(:,1)),1)];

        % Predict outer_test_x using opt_h
        predict = opt_h*outer_test_x';
        z = predict(end,:);
        predict = predict ./ z;

        % Root mean square error
        L2_dist = sqrt(sum((outer_test_y'-predict).^2));
        mse = mean(L2_dist);
        CV_error(i) = sqrt(mse);

        n_test = length(L2_dist);
        n_accurate = sum(L2_dist<50);
        accuracy(i) = n_accurate*100/n_test;

    %     predict = predict(1:2,:);
    %     outer_test_y = (outer_test_y(:,1:2))';
    %     figure(1);
    %     plot(predict(1,:),predict(2,:),'ro')
    %     hold on
    %     plot(outer_test_y(1,:),outer_test_y(2,:),'b*')
    %     temp = input("Press any key to continue");
    %     hold off
    end

    mean_CV_error = mean(CV_error)
    sd_CV_error = std(CV_error)
else
    disp('Too few correspondences')
end
