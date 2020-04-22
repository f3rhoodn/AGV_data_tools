# AGV_data_tools

To use these tools I will use Python and Matlab.

1) First, the 4DVirtualiz dataset should be downloaded from NAS. The data is simulated in two slightly different arrangments (with and without obstable). The data without obstacle is in folder "Vide" and the data with obstacle can be found in folder "Obstacle".
Structure of each folder (Obstacle for example):
```
  Obstacle------
      ----depth----
      ----depth_post---
      ----rgb------
      ----vehicle.csv----
      ----pedestrain.csv-----
      ----pedestrain2.csv------
```
      
Explanation:
  - depth: include raw depth files (.raw format).
  - dpeth_post: include processed depth files with additional noise to make it more realistic.
  - rgb: image file equivalent of the captured depth information.
  - vehicle: information regarding global coordinates (x,y,z) and orientations (yaw, pitch, roll) of the ego vehicle in the scene in a given time stamp.
  - pedestrain: there are 2 pedestrain in the scene. These csv files include their global coordinates and orientations.
  
2) In total, there are 12 sensors attached to the ego vehicle. There are two configurations of sensors for recording data. 
Method one and two. Image below shows the sensor configuration in the two methods.  
<img src="images/Capture_config.PNG" width="600">
Sensors "Front up" and "Rear up" are same for both configurations. 
Each sensor has its own coordinate and orientation values with respect to the center of AGV. All these values can be found in file: 

```
  positions_camera.txt
```
2) How to read data:

3) To visualize we can either use Python (pyPlot or Open3D recommended) or Matlab (pcshow)


