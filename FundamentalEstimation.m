%% Fundamental Matrix Estimation & Analysis
clc
close all
clear


cd('D:\MATLAB2021b\R2021b\bin\97113\Coursework1');

select = false;

%%%%%%%%%% load & show images %%%%%%%%%%
% \FD_Object_v1\FD_Object 23 & 24
% Stereo rectified 28&29\
% Epipole lines 23 & 24
% Vanishing points 38 & 39
I1 = 42; % 2836
I2 = 43; % 37
scalerfactor = 0.3;

%%%% FD set
FD1_ori1 = imread(['.\FD_Object_v1\FD_Object (', num2str(I1), ').jpg']);
% convert the image to a grayscale image
FD1_ori = imresize(FD1_ori1, scalerfactor);
FD1 = im2gray(FD1_ori);
% show the original image
figure(1)
imagesc(FD1_ori);

FD2_ori2 = imread(['.\FD_Object_v1\FD_Object (', num2str(I2), ').jpg']);
FD2_ori = imresize(FD2_ori2, scalerfactor);
% convert the image to a grayscale image
FD2 = im2gray(FD2_ori);
% show the original image
figure(2)
imagesc(FD2_ori);


%% Manual Detection

%%%%%%%%%% FD Estimation %%%%%%%%%%
%%%% Parameters
% number of points chosen
N = 16; % fundamental matrix needs at least 8.

if select
    %%%% Select corresponding points
    % store the corresponding points on the left image
    figure(1);
    hold on
    % preset matrix for the left images
    pointsfl = zeros(N, 2);
    for idx = 1:N
        pointsfl(idx, :) = ginput(1);
        plot(pointsfl(idx, 1), pointsfl(idx, 2), ...
            'rs', 'linewidth', 1, 'MarkerEdgeColor','r');
        text(pointsfl(idx, 1), pointsfl(idx, 2), num2str(idx), ...
                "HorizontalAlignment","center", ...
                "VerticalAlignment","middle");
    end
    hold off
    
    % store the corresponding points on the right image
    figure(2);
    hold on
    % preset matrix for the left images
    pointsfr = zeros(N, 2);
    for idx = 1:N
        pointsfr(idx, :) = ginput(1);
        plot(pointsfr(idx, 1), pointsfr(idx, 2), ...
            'rs', 'linewidth', 1, 'MarkerEdgeColor','g');
        text(pointsfr(idx, 1), pointsfr(idx, 2), num2str(idx), ...
                "HorizontalAlignment","center", ...
                "VerticalAlignment","middle");
    end
    hold off
    
    % save selected points
    save(['.\FD_Matrix\manual_Pairs', num2str(I1), '_', num2str(I2), '.mat'], "pointsfl", "pointsfr");
else
    load(['.\FD_Matrix\manual_Pairs', num2str(I1), '_', num2str(I2), '.mat']);
end

%%%% Estimate By self-function
[Fmanual_my, el_manual, er_manual] = fFDEstimation(pointsfl, pointsfr);
% Fmanual_my = fFDEstimation(pointsfl, pointsfr);
% store FD data
save(['.\FD_Matrix\manual_FD', num2str(I1), '_', num2str(I2), '.mat'], ...
      "Fmanual_my", "pointsfl", "pointsfr");

%%%% Estimate By 8 points method
[Fauto_8, inliers_8] = estimateFundamentalMatrix(pointsfl, pointsfr, 'method', 'norm8point');
[el_8p, er_8p] = fEpipolesEstimation(Fauto_8);

%%%% Visualise epipolar lines in images
fVisualEpipolar(FD1_ori, FD2_ori, Fmanual_my, pointsfl, pointsfr);
saveas(gcf, ['.\results\Manual_FD_epipole', num2str(I1), '_', num2str(I2), '.eps'], 'eps');
saveas(gcf, ['.\Results_figures\Manual_FD_epipole', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');

%%%% Epipole error
% M
el_err = sum((el_8p - el_manual).^2);
er_err = sum((er_8p - er_manual).^2);


%%%% F error
F_err = 0;
for i = 1:N
    tmp = [pointsfr(i, :), 1] * Fmanual_my * [pointsfl(i, :), 1].';
    F_err = F_err + tmp.^2;
end
F_err = F_err / N;


%%%% Visualise epipolar lines in images
fVisualEpipolar(FD1_ori, FD2_ori, Fauto_8, pointsfl, pointsfr);
saveas(gcf, ['.\results\Manual_FD8p_epipole', num2str(I1), '_', num2str(I2), '.epsc'], 'eps');
saveas(gcf, ['.\Results_figures\Manual_FD8p_epipole', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');

%%%%%%%%%% Reprojection Error Computation %%%%%%%%%%
MSE_normalmanual = fFDReproError(FD1_ori, pointsfl, pointsfr, Fmanual_my);

%% Automatic Detection

%%%%%%%%%% FD Estimation %%%%%%%%%%
%%% Keypoints Detection By Harris
% Key points detection by corner methods
pointsla = detectHarrisFeatures(FD1,'filtersize', 21);
pointsra = detectHarrisFeatures(FD2,'filtersize', 21);

% %%%% Keypoints Detection By SURF
% pointsla = detectSURFFeatures(FD1, 'MetricThreshold', 15000);
% pointsra = detectSURFFeatures(FD2, 'MetricThreshold', 15000);

% Extract the neighborhood features
[featurel, valid_pointsl] = extractFeatures(FD1, pointsla);
[featurer, valid_pointsr] = extractFeatures(FD2, pointsra);

% Match the features
indexPairs = matchFeatures(featurel, featurer);

% Retrieve the locations of the corresponding points for each image
matchedPl = valid_pointsl(indexPairs(:, 1), :);
matchedPr = valid_pointsr(indexPairs(:, 2), :);

%%%% Estimate By Least Median of Square method to find Inliers
% with inputs f [x y] coordinates
distancethre = 0.1;
[Fauto_RA, inliers_RA, status] = estimateFundamentalMatrix(matchedPl, matchedPr, ...
    'Method', 'RANSAC', ...
    'NumTrials', 2e5,'DistanceThreshold', distancethre, 'Confidence', 99.99);

% store FD data
save(['.\FD_Matrix\Auto_FD_Harris', num2str(I1), '_', num2str(I2), '.mat'], ...
      "Fauto_8", "matchedPl", "matchedPr", "inliers_RA");

% Visualize the Matched correspondances
figure
ax = axes;
showMatchedFeatures(FD1_ori, FD2_ori, matchedPl(inliers_RA), matchedPr(inliers_RA),'montage','Parent',ax);

saveas(gcf, ['.\results\Harris_FD_correspondeces', num2str(I1), '_', num2str(I2), '.eps'], 'epsc');
saveas(gcf, ['.\Results_figures\Harris_FD_correspondeces', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');

% Epipoles estimation
[el_RA, er_RA] = fEpipolesEstimation(Fauto_RA);

% Inlier correspondances
inlierpl = matchedPl.Location;
inlierpr = matchedPr.Location;

% Calculated the quality
Quantity_Harris = length(inliers_RA);
Quality_Harris = sum(inliers_RA) / Quantity_Harris;


%%%% Keypoints Detection By SURF
% pointsla = detectSURFFeatures(FD1, 'MetricThreshold', 8000, 'NumOctaves', 3, 'NumScaleLevels', 6);
% pointsra = detectSURFFeatures(FD2, 'MetricThreshold', 8000, 'NumOctaves', 3, 'NumScaleLevels', 6);
pointsla = detectSURFFeatures(FD1, 'MetricThreshold', 8000, 'NumScaleLevels', 3);
pointsra = detectSURFFeatures(FD2, 'MetricThreshold', 8000, 'NumScaleLevels', 3);

% Extract the neighborhood features
[featurel, valid_pointsl] = extractFeatures(FD1, pointsla);
[featurer, valid_pointsr] = extractFeatures(FD2, pointsra);

% Match the features
indexPairs = matchFeatures(featurel, featurer);

% Retrieve the locations of the corresponding points for each image
matchedPl = valid_pointsl(indexPairs(:, 1), :);
matchedPr = valid_pointsr(indexPairs(:, 2), :);

%%%% Estimate By Least Median of Square method to find Inliers
% with inputs f [x y] coordinates
distancethre = 0.1;
[Fauto_RA, inliers_RA, status] = estimateFundamentalMatrix(matchedPl, matchedPr, ...
    'Method', 'RANSAC', ...
    'NumTrials', 2e5,'DistanceThreshold', distancethre, 'Confidence', 99.99);

% store FD data
save(['.\FD_Matrix\Auto_FD_SURF', num2str(I1), '_', num2str(I2), '.mat'], ...
      "Fauto_8", "matchedPl", "matchedPr", "inliers_RA");

% Visualize the Matched correspondances
figure
ax = axes;
showMatchedFeatures(FD1_ori, FD2_ori, matchedPl(inliers_RA), matchedPr(inliers_RA),'montage','Parent',ax);
saveas(gcf, ['.\results\SURF_FD_correspondeces', num2str(I1), '_', num2str(I2), '.eps'], 'epsc');
saveas(gcf, ['.\Results_figures\SURF_FD_correspondeces', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');

% Calculated the quality
Quantity_SURF = length(inliers_RA);
Quality_SURF = sum(inliers_RA) / Quantity_SURF;

