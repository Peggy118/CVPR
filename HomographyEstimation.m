%% Homography Matrix Estimation & Analysis
clc
clear
close all


select = false;
estimation = true;


%%%%%%%%%% load & show images %%%%%%%%%%
%%%% Rotation 4 & 5, 5&8
%%%% Scaling  1 & 3 (scaling factor s = 1.5) 5&6
%%%% Rotation & Scaling 7 & 8
I1 = 1;
I2 = 3;
scalerfactor = 0.3;

%%%% HG set
HG1_ori = imread(['.\HG\HG (', num2str(I1), ').jpg']);
% convert the image to a grayscale image
HG1_ori = imresize(HG1_ori, scalerfactor);
HG1 = im2gray(HG1_ori);
% show the original image
figure(1)
imagesc(HG1_ori);

HG2_ori = imread(['.\HG\HG (', num2str(I2), ').jpg']);
% convert the image to a grayscale image
HG2_ori = imresize(HG2_ori, scalerfactor);
HG2 = im2gray(HG2_ori);
% show the original image
figure(2)
imagesc(HG2_ori);

%% Manual Detection

%%%%%%%%%% HG Estimation %%%%%%%%%%
%%%% Parameters
% number of points chosen
N = 16; % fundamental matrix needs at least 8.

if select
    %%%% Select corresponding points
    % store the corresponding points on the left image
    figure(1);
    hold on
    % preset matrix for the left images
    pointshl = zeros(N, 2);
    for idx = 1:N
        pointshl(idx, :) = ginput(1);
        plot(pointshl(idx, 1), pointshl(idx, 2), ...
                'rs', 'linewidth', 1, 'MarkerEdgeColor','r');
        text(pointshl(idx, 1), pointshl(idx, 2), num2str(idx), ...
                    "HorizontalAlignment","center", ...
                    "VerticalAlignment","middle");
    end
    hold off
    
    % store the corresponding points on the right image
    figure(2);
    hold on
    % preset matrix for the right images
    pointshr = zeros(N, 2);
    for idx = 1:N
        pointshr(idx, :) = ginput(1);
        plot(pointshr(idx, 1), pointshr(idx, 2), ...
                'rs', 'linewidth', 1, 'MarkerEdgeColor','g');
        text(pointshr(idx, 1), pointshr(idx, 2), num2str(idx), ...
                    "HorizontalAlignment","center", ...
                    "VerticalAlignment","middle");
    end
    hold off
    
    % save selected points
    save(['.\HG_Matrix\manual_Pairs', num2str(I1), '_', num2str(I2), '.mat'], "pointshl", "pointshr");
else
    load(['.\HG_Matrix\manual_Pairs', num2str(I1), '_', num2str(I2), '.mat']);
end

%%%% Estimated by self function
Hmanual_my = fHGEstimation(pointshl, pointshr);
[MSE_normanual, project2to1] = fReproError(pointshl, pointshr, Hmanual_my);
save(['.\HG_Matrix\manual_HG', num2str(I1), '_', num2str(I2), '.mat'], "Hmanual_my", "MSE_normanual");

%%%% Reproject points
% [Htrans, inliers] = estimateGeometricTransform2D(matchedPr, matchedPl, 'projective');
fReCorrespondencesPlot(HG1_ori, HG2_ori, pointshl, pointshr, project2to1(:, 1:2));
saveas(gcf, ['.\Results_figures\manual_HG_Reproject', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');


%% Automatic Detection

if estimation
    %%%%%%%%%% HG Estimation by Harris (rotation or scale) %%%%%%%%%%
    % Key points detection
    pointsla = detectHarrisFeatures(HG1, 'filtersize', 51);
    pointsra = detectHarrisFeatures(HG2, 'filtersize', 51);
    % Extract features
    [featurel, vptsl] = extractFeatures(HG1, pointsla);
    [featurer, vptsr] = extractFeatures(HG2, pointsra);
    
    % Retrieve the locations of matched points
    indexPairs2 = matchFeatures(featurel, featurer);
    matchedPl = vptsl(indexPairs2(:, 1), :);
    matchedPr = vptsr(indexPairs2(:, 2), :);
    
    %%%% Estimate under projective
    % projection from right image to left image
    [Htrans, inliers] = estimateGeometricTransform2D(matchedPr, matchedPl, 'projective');
    % Calculated the quality & quantity
    Quantit_Harris = size(matchedPl.Location, 1);
    Quality_Harris = size(matchedPl.Location(inliers), 1) / Quantit_Harris;
    
    % homography matrix of (projection of left to right)
    Hauto = (Htrans.T).';
    % display
    Hauto
    % store FD data
    save(['.\HG_Matrix\Auto_HG_Harris', num2str(I1), '_', num2str(I2), '.mat'], ...
          "Hauto", "inliers", "matchedPl", "matchedPr");
    
    % Visualise the matching points excluding outliers
    figure
    ax = axes;
%     showMatchedFeatures(HG1, HG2, matchedPl(inliers, :), matchedPr(inliers, :));
%     title(['Automatic Correspondences Detection between HG', num2str(I1), ' and HG', num2str(I2), 'by Harris']);
    showMatchedFeatures(HG1_ori, HG2_ori, matchedPl, matchedPr,'montage','Parent',ax);
%     title(ax, ['Automatic Correspondences of HG', num2str(I1), ' and ', num2str(I2)]'fontsize',20);
    saveas(gcf, ['.\results\Auto_HG_Harris', num2str(I1), '_', num2str(I2), '.eps'], 'eps');
    saveas(gcf, ['.\Results_figures\Correpondences_HG_Harris', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');

    %%%%%%%%%% HG Estimation by SURF (rotation or scale) %%%%%%%%%%
    % % Key points detection - Scaling
    % pointsla = detectSURFFeatures(HG1, 'metricthreshold', 12000, 'numoctaves',3, 'numscalelevels',6);
    % pointsra = detectSURFFeatures(HG2, 'metricthreshold', 12000, 'numoctaves',3, 'numscalelevels',6);
    
    % Key points detection - Rotation
    pointsla = detectSURFFeatures(HG1, 'metricthreshold', 8000, 'numoctaves',3, 'numscalelevels',4);
    pointsra = detectSURFFeatures(HG2, 'metricthreshold', 8000, 'numoctaves',3, 'numscalelevels',4);
    
    % Extract features
    [featurel, vptsl] = extractFeatures(HG1, pointsla);
    [featurer, vptsr] = extractFeatures(HG2, pointsra);
    
    % Retrieve the locations of matched points
    indexPairs2 = matchFeatures(featurel, featurer);
    matchedPl = vptsl(indexPairs2(:, 1), :);
    matchedPr = vptsr(indexPairs2(:, 2), :);
    
    %%%% Estimate under projective
    % projection from right image to left image
    [Htrans, inliers] = estimateGeometricTransform2D(matchedPr, matchedPl, 'projective');
    % Calculated the quality & quantity
    Quantit_SURF = size(matchedPl.Location, 1);
    Quality_SURF = size(matchedPl.Location(inliers), 1) / Quantit_SURF;
    
    
    % homography matrix of (projection of left to right)
    Hauto = (Htrans.T).';
    % display
    Hauto

    % store FD data
    save(['.\HG_Matrix\Auto_HG_SURF', num2str(I1), '_', num2str(I2), '.mat'], ...
          "Hauto", "inliers", "matchedPl", "matchedPr");
    
    % Visualise the matching points excluding outliers
    figure
    ax = axes;
%     showMatchedFeatures(HG1, HG2, matchedPl(inliers, :), matchedPr(inliers, :));
%    title(['Automatic Correspondences Detection between HG', num2str(I1), ' and HG', num2str(I2), 'by SURF']);
    showMatchedFeatures(HG1_ori, HG2_ori, matchedPl, matchedPr,'montage','Parent',ax);
%     title(ax, ['Automatic Correspondences of HG', num2str(I1), ' and ', num2str(I2)], 'fontsize',18);
    saveas(gcf, ['.\results\Auto_HG_SURF', num2str(I1), '_', num2str(I2), '.eps'], 'epsc');
    saveas(gcf, ['.\Results_figures\Correspondences_HG_SURF', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');

    %%%% Reproject points
    [MSE_normauto, project2to1] = fReproError(pointshl, pointshr, Hauto);
    fReCorrespondencesPlot(HG1_ori, HG2_ori, pointshl, pointshr, project2to1(:, 1:2));
    saveas(gcf, ['.\Results_figures\Auto_HG_Reproject', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');



    %%%% Estimate by self function
    % Inlier correspondances
    inlierpl = matchedPl.Location(inliers, :);
    inlierpr = matchedPr.Location(inliers, :);
    Hauto_my = fHGEstimation(double(inlierpl), ...
                             double(inlierpr));
    
    
%     %%%%%%%%%% Reconstruction Image %%%%%%%%%%
%     outputview = imref2d(size(HG1));
%     % Reconstruct the left image2 according to right image
%     HGrecon = imwarp(HG2, Htrans, 'OutputView', outputview);
%     figure
%     imagesc(HGrecon);
else
    load(['.\HG_Matrix\Auto_HG_SURF', num2str(I1), '_', num2str(I2), '.mat']);

    % Inlier correspondances
    inlierpl = matchedPl.Location(inliers, :);
    inlierpr = matchedPr.Location(inliers, :);

    % Calculated the quality & quantity
    Quantit_SURF = size(matchedPl.Location, 1);
    Quality_SURF = size(matchedPl.Location(inliers), 1) / Quantit_SURF;

end

%%
%%%%%%%%%% Reprojection Error Computation %%%%%%%%%%
% form the 3D points
z_axis = ones(length(inlierpl(:, 1)), 1);
inlier_pl = [inlierpl z_axis];
inlier_pr = [inlierpr z_axis];

%%%% Project of HG2 to HG1
project2to1 = zeros(length(z_axis), 3);

inlier_prtr = inlier_pr.';

for c = 1:length(z_axis)
    project2to1(c, 1) = Hauto(1, :) * inlier_prtr(:, c);
    project2to1(c, 2) = Hauto(2, :) * inlier_prtr(:, c);
    project2to1(c, 3) = Hauto(3, :) * inlier_prtr(:, c);
end

%%%% MSE calculation
% error from reprojection of 2 to 1
MSE2to1 = immse(double(inlier_pl), project2to1);
% % jpg image pixels
% total_num_of_pixels = 65.535 * 65.535;
% % Normalised MSE for each pixel
% MSE_normal2to1 = MSE2to1 ./ total_num_of_pixels;



%% Analysis of Tolerances of Outliers

%%%% Criteria
% when the MSE > criteria, the estimation is untolerable
criteria = 100;

%%%% Estimate by self function
% Obtain the number of outliers
Noutliers = length(inliers) - size(inlierpl, 1);
% Set the interval
interval = 1;
% Add Ninterval outliers at one loop
Maxloops = ceil(Noutliers / interval) - 1;
% Preset the MMSE matrix
MMSE_tol = zeros(Maxloops+1, 1);
Noutliers_tol = zeros(Maxloops+1, 1);
MMSE_tol(1) = MSE2to1; % comparison

% Obtain indices of all outliers
Outliers = find(inliers == false);

% Copy the inliers
InOutliers = inliers;

%%%% Computation reprojection error for all outliers
for i = 1:Maxloops
    %%%% Change logical value in the indices of inliers to include outliers
    % Add outliers
    InOutliers(Outliers(1:i*interval)) = true;

    % Correspondances including some outliers
    pointsl_out = double(matchedPl.Location(InOutliers, :));
    pointsr_out = double(matchedPr.Location(InOutliers, :));
    % Obtain the ratio of current outliers added and the total number of
    % correspondences
    Noutliers_tol(i+1) = i*interval / length(pointsr_out) * 100;

    %%%% Homograph estimation
    Hmatrix = fHGEstimation(double(pointsl_out), ...
                            double(pointsr_out));

    %%%% Reprojection Error Computation
    [MMSE_tol(i+1), ~] = fReproError(pointsl_out, pointsr_out, Hmatrix);
end



%% Functions

function eigV = fSVDminimum(A)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   - A (matrix)
% Output:
%   - eigV (vector): eigvector associated with the smallest eigenvalues
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if x == y % A is a square matrix
    % SVD
    [~, ~, eigVect] = svd(A' * A);
    % eigVect is ordered associated with eigenvalues
    eigV = eigVect(:, end);
% end
end


function H = fHGEstimation(pl, pr)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function:
% To find the homography matrix in planar view by solving xL = H*xR
% H matrix of the projection of right image pr to left image pl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   - pl (Nx2 Doubles): N points selected on the left images
%   - pr (Nx2 Doubles): N points selected on the right images
% Output:
%   - H (3x3 Complexs): Homography matrix between the 2 images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Obtain the number of correspondances
[N, ~] = size(pl);

% Form the A matrix
A = zeros(2*N, 9); % N correspondances
for i = 1:size(pl, 1)
    xl = pl(i, 1);
    yl = pl(i, 2);
    xr = pr(i, 1);
    yr = pr(i, 2);
    % odd rows
    A(2*i-1, :) = [xr, yr, 1, 0, 0, 0, -xl*xr, -xl*yr, -xl];
    % even rows
    A(2*i, :) = [0, 0, 0, xr, yr, 1, -yl*xr, -yl*yr, -yl];
end

% Compute transformation
hvec = fSVDminimum(A);

% Normalise and rearrange the vectorized f
hvec = hvec ./ hvec(end);
H = [hvec(1), hvec(2), hvec(3); ...
     hvec(4), hvec(5), hvec(6); ...
     hvec(7), hvec(8), hvec(9)];

end


function [MSE2to1, project2to1] = fReproError(pl, pr, Hmatrix)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function:
% Computation of reprojection error (from pr to pl)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   - pl (Nx2 Doubles): N points selected on the left images
%   - pr (Nx2 Doubles): N points selected on the right images
%   - H (3x3 Complexs): Homography matrix between the 2 images
% Output:
%   - MSE_normal2to1 (Double): Normalised MMSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% form the 3D points
z_axis = ones(length(pl(:, 1)), 1);
pointl = [pl z_axis];
pointr = [pr z_axis];

%%%% Project of HG2 to HG1
project2to1 = zeros(length(z_axis), 3);

pointrtr = pointr.';

for c = 1:length(z_axis)
    project2to1(c, 1) = Hmatrix(1, :) * pointrtr(:, c);
    project2to1(c, 2) = Hmatrix(2, :) * pointrtr(:, c);
    project2to1(c, 3) = Hmatrix(3, :) * pointrtr(:, c);
end

%%%% MSE calculation
% error from reprojection of right to left
MSE2to1 = immse(double(pointl), project2to1);
MSE2to1 = MSE2to1;


end