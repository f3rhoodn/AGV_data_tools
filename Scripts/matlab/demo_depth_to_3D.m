
RGB_file = '.\rgbSample.png';
depth_file = '.\depthSample.raw';

f = figure;
%resize and adjust the window in center of screen
scrsz = get(groot,'ScreenSize');
x0=(0.1)*scrsz(3);
y0=(0.17)*scrsz(4);
width=1024;
height=400;
set(gcf,'position',[x0,y0,width,height]);
movegui(f,'center');

FID=fopen(depth_file,'r');
DepthImage = fread(FID,500*290*4,'float');
fclose(FID);
depth_data = reshape(DepthImage, 500,290);
%%%%%%%%%%%%% Point Cloud %%%%%%%%%%%%%%%%%%%%%%%%%%%%
point_cloud_fig = subplot(2,2,[3 4]);
K = [144.337567297406 0 250;0 350.060966544099 145;0 0 1];
%convert depth values to the 3d point cloud coordinates
for x=1 : 500
    for y =1: 290
        d = depth_data(x,y);
        P{x,y} = pixel2pts3d(K,x,y,d);
    end
end
%make the points sutiable for point cloud structure format
xyzPoints = zeros(500,290,3);
%PointsColor = uint8(zeros(500,290,3));
all_pts=[];
for x=1 : 500
    for y =1: 290
        points = P{x,y};
        
        xyzPoints(x,y,1)= points(1,1);
        xyzPoints(x,y,2)= points(2,1);
        xyzPoints(x,y,3)= points(3,1);
    end
end

pCloud = pointCloud(xyzPoints);
%%%%% Downsampling
%ptCloudOut = pcdownsample(pCloud,'random',0.5);
gridSize = 0.1;
ptCloudOut = pcdownsample(pCloud, 'gridAverage', gridSize);
%pCloud.Color=PointsColor;
pcshow(ptCloudOut);
set(gca,'color','black');
%set view
az = 0;
el = -65;
view(az, el);
zoom(3);
title('Point Cloud');
colormap(point_cloud_fig,jet);

%%%%%%%%%%%%% RGB %%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,1);
imshow(RGB_file);
%hold on;
title('RGB stream');
%%%%%%%%%%%%% Depth %%%%%%%%%%%%%%%%%%%%%%%%%%%%
depth_data_for_gray_scale = depth_data';

gray_converter = mat2gray(depth_data_for_gray_scale);
subplot(2,2,2);
imshow(gray_converter);
% colormap jet;
%hold on;
title('Depth Map');

