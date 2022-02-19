%% Camera Calibration
clc
close all
clear

cd('D:\MATLAB2021b\R2021b\bin\97113\Coursework1');
select = false;

%%%%%%%%%% load & show images %%%%%%%%%%
% rectification 
% disparity map 30 & 31
I1 = 6;  %26  30
I2 = 7;  %27  31

%%%% FD set
FD1_ori = imread(['.\FD_Object_v1\FD_Object (', num2str(I1), ').jpg']);
FD1_ori = imresize(FD1_ori, 0.3);
% % convert the image to a grayscale image
% FD1 = im2gray(FD1_ori);
% show the original image
figure(1)
imagesc(FD1_ori);
FD2_ori = imread(['.\FD_Object_v1\FD_Object (', num2str(I2), ').jpg']);
FD2_ori = imresize(FD2_ori, 0.3);
% % convert the image to a grayscale image
% FD2 = im2gray(FD2_ori);
% show the original image
figure(2)
imagesc(FD2_ori);


%% Rectify stereo Images
%%%% Obtain the transformation matrix for camera 2 from camera1

% obtain inlier points
load(['.\FD_Matrix\Auto_FD_Harris', num2str(I1), '_', num2str(I2), '.mat']);

% distancethre = 0.1;
% [F, inliers, status] = estimateFundamentalMatrix(matchedPl, matchedPr, ...
%     'Method', 'RANSAC', ...
%     'NumTrials', 2e5,'DistanceThreshold', distancethre, 'Confidence', 99.99);
pointsl = matchedPl;
pointsr = matchedPr;
% 
% % %%% Check if epipoles are in the images
% if status ~= 0 || isEpipoleInImage(F, size(FD1_ori)) ...
%   || isEpipoleInImage(F', size(FD2_ori))
%   error(['Either not enough matching points were found or '...
%          'the epipoles are inside the images. You may need to '...
%          'inspect and improve the quality of detected features ',...
%          'and/or improve the quality of your images.']);
% end

%%
% % obtain inlier points
% load(['.\FD_Matrix\Auto_FD_Harris', num2str(I1), '_', num2str(I2), '.mat']);
% load(['.\FD_Matrix\manual_FD', num2str(I1), '_', num2str(I2), '.mat']);
F = Fauto_8;
% pointsl = matchedPl;
% pointsr = matchedPr;
% 
% %%%% Check if epipoles are in the images
% if isEpipoleInImage(F, size(FD1_ori)) ...
%   || isEpipoleInImage(F', size(FD2_ori))
%   error(['Either not enough matching points were found or '...
%          'the epipoles are inside the images. You may need to '...
%          'inspect and improve the quality of detected features ',...
%          'and/or improve the quality of your images.']);
% end

%%
% Get estimated epipoles for both images
[el, er] = fEpipolesEstimation(F);

% obtain the size of image
imagesize = size(FD1_ori);

%%%% Uncalibrated stereo rectification
% returns projective transformations for rectifying stereo images. 
% This function does not require either intrinsic or extrinsic camera parameters.
% Tx - describing the projective transformations for imagex
% An epipole in the images may leas to undesired distortion
[T1, T2] = estimateUncalibratedRectification(F, pointsl, pointsr, imagesize);
% Tform1 = rigid3d(T1);
Tform1 = projective2d(T1);
Tform2 = projective2d(T2);

% %%%% Form the stereo parameters
% stereoParams = stereoParameters(cameraParameters3, cameraParameters3, Tform1, Tform2);

%%%% Rectify images by applying projective transformations matices
[FD1_rect, FD2_rect] = rectifyStereoImages(FD1_ori, FD2_ori, Tform1, Tform2, 'OutputView','valid');
[FD1_rectfull, FD2_rectfull] = rectifyStereoImages(FD1_ori, FD2_ori, Tform1, Tform2, 'OutputView', 'full');
figure(3)
imshow(stereoAnaglyph(FD1_rectfull, FD2_rectfull));


%%

%%%%%%%%%% FD Estimation %%%%%%%%%%
%%%% Parameters
% number of points chosen
N = 16; % fundamental matrix needs at least 8.

if select
    %%%% Select corresponding points
    % store the corresponding points on the left image
    figure
    imagesc(FD1_rectfull);
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
    figure
    imagesc(FD2_rectfull);
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
    save(['.\FD_Matrix\manual_RectPairs', num2str(I1), '_', num2str(I2), '.mat'], "pointsfl", "pointsfr");
else
    load(['.\FD_Matrix\manual_RectPairs', num2str(I1), '_', num2str(I2), '.mat']);
end

%%
%%%% Estimate By self-function
% [Frect, ~] = estimateFundamentalMatrix(pointsfl, pointsfr, 'method', 'norm8point');
Frect = zeros(3,3);
Frect(2,3) = -1;
Frect(3,2) = 1;

%%%% Visualise epipolar lines in images
fVisualEpipolar(FD1_rectfull, FD2_rectfull, Frect, pointsfl, pointsfr);
saveas(gcf, ['.\results\Rectified_FD', num2str(I1), '_', num2str(I2), '.eps'], 'epsc');
saveas(gcf, ['.\Results_figures\Rectified_FD', num2str(I1), '_', num2str(I2), '.jpg'], 'jpg');
%% Depth Estimation

%%%% Computation of the distance (disparity) of each pixel in the left image to the
%%%% corresponding pixel in the right image.
% convert the image to a grayscale image
FD1 = rgb2gray(FD1_rect);
FD2 = rgb2gray(FD2_rect);

disparityRange = [-24 24];
disparityMap = disparityBM(FD1, FD2,'DisparityRange',disparityRange,'uniquenessthreshold', 25);
% ...
%     'DisparityRange', disparityRange, ...
%     'uniquenessthreshold',30);

%%%% Visual the disparity map
figure
imshow(disparityMap, disparityRange);
title('Disparity Map');
colormap jet
colorbar

saveas(gcf, ['results/Disparity_Map', num2str(I1), '_', num2str(I2), '.eps'],'epsc');
saveas(gcf, ['Results_figures/Disparity_Map', num2str(I1), '_', num2str(I2), '.jpg'],'jpg');

% %% 3D Reconstruction
% points3D = reconstructScene(disparityMap, stereoParams);
% 
% % Convert to meters and create a pointCloud object
% points3D = points3D ./ 1000;
% ptCloud = pointCloud(points3D, 'Color', frameLeftRect);
% 
% % Create a streaming point cloud viewer
% player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
%     'VerticalAxisDir', 'down');
% 
% % Visualize the point cloud
% view(player3D, ptCloud);
