def pcTranslation(points, th_yaw, th_pitch, th_roll, c)


# negate the angle
Ryaw = [cos(deg2rad(-th_yaw)) - sin(deg2rad(-th_yaw)) 0 0
        sin(deg2rad(-th_yaw)) cos(deg2rad(-th_yaw)) 0 0
        0 0 1 0
        0 0 0 1]  # Rotation around z axis
Rpitch = [cos(deg2rad(-th_pitch)) 0 sin(deg2rad(-th_pitch)) 0
          0 1 0 0
          - sin(deg2rad(-th_pitch)) 0 cos(deg2rad(-th_pitch)) 0
          0 0 0 1]  # Rotation around y axis
Rroll = [1 0 0 0
         0 cos(deg2rad(-th_roll)) - sin(deg2rad(-th_roll)) 0
         0 sin(deg2rad(-th_roll)) cos(deg2rad(-th_roll)) 0
         0 0 0 1]   # Rotation around x axis

R_ = Rroll*Rpitch*Ryaw

# negate the translation
C_ = [1  0   0   c(1)
      0  1   0   c(2)
      0  0   1   c(3)
      0  0   0   1]
T_ = C_*R_




coord_all_curr = []
for i = 1:
    size(points, 1):
    coord_1 = points(i, 1)
    coord_2 = points(i, 2)
    coord_3 = points(i, 3)
    coord_4 = 1
    a = T_*[coord_1
            coord_2
            coord_3
            coord_4]
    a = a'
    coord_all_curr = vertcat(coord_all_curr, a)


return pointCloud(coord_all_curr(:, 1: 3))
