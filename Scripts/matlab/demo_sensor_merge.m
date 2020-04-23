%%%%%%%%%%%%% load depth data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
depths{1} = '.\frontUp.raw';   %depth from left sensors
depths{2} = '.\frontLeft.raw'; %depth from center sensors
depths{3} = '.\fontCenter.raw'; %depth from front up sensors

for i=1:3
    FID=fopen(depths{i},'r');
    DepthImage = fread(FID,500*290*4,'float');
    fclose(FID);
    alldepths{i} = reshape(DepthImage, 500,290);
end

%%%%%%%%%%%%% Point Cloud %%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = [144.337567297406 0 250;0 350.060966544099 145;0 0 1];
%convert depth values to the 3d point cloud coordinates
for i=1:3
    xyzPoints{i} = zeros(500,290,3);
    depth_data = alldepths{i};
    for x=1 : 500
        for y =1: 290
            d = depth_data(x,y);
            P{x,y} = pixel2pts3d(K,x,y,d);
        end
    end
    allP{i} = P;
end

for i=1:3
    P = allP{i};
    for x=1 : 500
        for y =1: 290
            points = P{x,y};
            Points(x,y,1)= points(1,1);
            Points(x,y,2)= points(2,1);
            Points(x,y,3)= points(3,1);
        end
    end
    ptClouds{i} = pointCloud(Points);
    gridSize = 0.1;
    ptClouds{i} = pcdownsample(ptClouds{i}, 'gridAverage', gridSize); %Downsampling
end

%%%%%%%%%%%%% Merge %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sensors
%1:  front center up        ======> same in both methods
%2:  front left             ======> method method 2
%3:  front left - left      ======> method method 1
%4:  front left - right     ======> method method 1
%5:  front left - center    ======> method method 1
%6:  front right            ======> method method 2
%7:  rear center up         ======> same in both methods
%8:  rear left              ======> method method 2
%9:  rear right - right     ======> method method 1
%10: rear right - center    ======> method method 1
%11: rear right - left      ======> method method 1
%12: rear right             ======> method method 2

%sensors used: 1, 3, 5
yawPitchRollVals = [0,0,30.5;
    0,-30,-17.5;
    0,45,-17.5;
    0,-30,-17.5;
    0,120,-17.5;
    0,-120,-17.5;
    0,180,-67.5;
    0,60,-17.5;
    0,150,-17.5;
    0,-60,-17.5;
    0,-135,-17.5;
    0,150,-17.5];

%x,y,z
sensorPositions = [0,-1.805,2;
    -0.645,0.118,2.1;
    -0.645,0.118,2.1;
    -0.645,0.118,2.1;
    -0.645,0.118,2.1;
    0.645,0.118,2.1;
    0,-1.755,-0.55;
    -0.645,0.118,-0.6;
    0.645,0.118,-0.6;
    0.645,0.118,-0.6;
    0.645,0.118,-0.6;
    0.645,0.118,-0.6];

%%%%%%% sensor 1 translation 
sensorID = 1;
th_roll = yawPitchRollVals(sensorID,3);
c = [0 0 0]';
ptClouds{1} = pcTranslation(ptClouds{1},0,0,th_roll, c);
% translate to vehicle center coordinates
c = sensorPositions(sensorID,:)';
ptClouds{1} = pcTranslation(ptClouds{1},0,0,0, c);

%%%%%%% sensor 3 translation 
sensorID = 3;
c = [0 0 0]';
th_roll = yawPitchRollVals(sensorID,3);
ptClouds{2} = pcTranslation(ptClouds{2},0,0,th_roll, c);
th_pitch =  yawPitchRollVals(sensorID,2);
ptClouds{2} = pcTranslation(ptClouds{2},0,th_pitch,0, c);
% translate to vehicle center coordinates
c = sensorPositions(sensorID,:)';
ptClouds{2} = pcTranslation(ptClouds{2},0,0,0, c);

%%%%%%% sensor 5 translation 
sensorID = 5;
c = [0 0 0]';
th_roll = yawPitchRollVals(sensorID,3);
ptClouds{3} = pcTranslation(ptClouds{3},0,0,th_roll, c);
th_pitch =  yawPitchRollVals(sensorID,2);
ptClouds{3} = pcTranslation(ptClouds{3},0,th_pitch,0, c);
% translate to vehicle center coordinates
c = sensorPositions(sensorID,:)';
ptClouds{3} = pcTranslation(ptClouds{3},0,0,0, c);

tmp1 = ptClouds{1}.Location;
tmp2 = ptClouds{2}.Location;
tmp3 = ptClouds{3}.Location;
tmp4 = vertcat(tmp1,tmp2,tmp3);
finalPtCloud = pointCloud(tmp4);

pcshow(finalPtCloud);
