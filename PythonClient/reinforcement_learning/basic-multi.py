import airsim
import os
from PIL import Image
import numpy as np
import time
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.simSetSegmentationObjectID("[\w]*", 0, True)
client.simSetSegmentationObjectID("DroneFollower1", 10, True) #[29, 26, 199]
client.simSetSegmentationObjectID("DroneFollower2", 20, True)  # [146, 52, 70]      
client.simSetSegmentationObjectID("detector_1", 30, True) #[226, 149, 143], pos:
client.simSetSegmentationObjectID("detector_6", 40, True) #[151, 126, 171, pos:



client.enableApiControl(True, "DroneFollower1")
client.enableApiControl(True, "DroneFollower2")

client.armDisarm(True, "DroneFollower1")
client.armDisarm(True, "DroneFollower2")

client.takeoffAsync(vehicle_name="DroneFollower1").join()
client.moveToPositionAsync(5, 0, -5, 10,vehicle_name="DroneFollower1").join()

client.takeoffAsync(vehicle_name="DroneFollower2").join()
client.moveToPositionAsync(-5, 0, -5, 10,vehicle_name="DroneFollower2").join()





image_request_1 = airsim.ImageRequest("FixedCamera1", airsim.ImageType.Segmentation,False, False)
responses = client.simGetImages([image_request_1], external=True)

response = responses[0]
img = np.fromstring(response.image_data_uint8, dtype=np.uint8)
img_rgb = img.reshape(response.height, response.width, 3)
image = Image.fromarray(img_rgb)
img_seg = np.array(image.resize((256,256)))
cv2.imwrite("s.png", img_seg)







image_request_2 = airsim.ImageRequest("FixedCamera1", airsim.ImageType.Scene,False, False)
responses = client.simGetImages([image_request], external=True)

response = responses[0]
img = np.fromstring(response.image_data_uint8, dtype=np.uint8)
img_rgb = img.reshape(response.height, response.width, 3)
image = Image.fromarray(img_rgb)
img = np.array(image.resize((response.height, response.width)))
cv2.imwrite("i.png", img)








client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


client.enableApiControl(True, "DroneLeader")
client.armDisarm(True, "DroneLeader")
client.takeoffAsync(vehicle_name="DroneLeader").join()
client.moveToPositionAsync(0, 0, -100, 10,vehicle_name="DroneLeader").join()

success= client.simSetSegmentationObjectID("[\w]*", 0, True)
print(success)

success = client.simSetSegmentationObjectID("DroneFollower1", 77)
print(success)




cam_info = client.simGetCameraInfo("FixedCamera1", external=True)
print(cam_info)

image_request = airsim.ImageRequest("FixedCamera1", airsim.ImageType.Scene,False, False)
responses = client.simGetImages([image_request], external=True)

response = responses[0]
img = np.fromstring(response.image_data_uint8, dtype=np.uint8)
img_rgb = img.reshape(response.height, response.width, 3)
image = Image.fromarray(img_rgb)
im_final = np.array(image.resize((1080, 1080)))
cv2.imwrite("s.png", im_final)


client.simSetSegmentationObjectID("[\w]*", 0, True)
client.simSetSegmentationObjectID("DroneFollower1", 10, True) #[29, 26, 199]
client.simSetSegmentationObjectID("DroneFollower2", 20, True)  # [146, 52, 70]      
client.simSetSegmentationObjectID("detector_1", 30, True) #[226, 149, 143], pos:
client.simSetSegmentationObjectID("detector_6", 40, True) #[151, 126, 171, pos:









image_request = airsim.ImageRequest(3, airsim.ImageType.Segmentation, False, False)

responses = client.simGetImages([image_request],camera_name="FixedCamera1",external=True)


response = responses[0]
img = np.fromstring(response.image_data_uint8, dtype=np.uint8)
img_rgb = img.reshape(response.height, response.width, 3)
image = Image.fromarray(img_rgb)
im_final = np.array(image.resize((128, 128)))
cv2.imwrite("s.png", im_final)


client.enableApiControl(True, "DroneFollower1")
client.enableApiControl(True, "DroneFollower2")

client.armDisarm(True, "DroneLeader")
client.armDisarm(True, "DroneFollower1")
client.armDisarm(True, "DroneFollower2")

client.takeoffAsync(vehicle_name="DroneLeader").join()
client.takeoffAsync(vehicle_name="DroneFollower1").join()
client.takeoffAsync(vehicle_name="DroneFollower2").join()



client.moveToPositionAsync(0, 0, -100, 10,vehicle_name="DroneLeader").join()
client.moveToPositionAsync(2, 0, -5, 10,vehicle_name="DroneFollower1").join()
client.moveToPositionAsync(0, 0, -75, 10,vehicle_name="DroneFollower2").join()

lt=[]
for i in range(1,201):
    lt.append([0,-i])

for i in range(1,101):
    lt.append([-i,-200])

for i in range(1,301):
    lt.append([-100,-200-i])

for i in range(1,126):
    lt.append([-100-i,-500])

for i in range(1,401):
    lt.append([-225,-500-i])


for i in range(len(lt)):
    client.moveToPositionAsync(lt[i][0], lt[i][1], -100, 10,vehicle_name="Drone1").join()
    client.hoverAsync(vehicle_name="Drone1").join()
    print(client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position)


exit()

success = client.simSetSegmentationObjectID("base_1", 54);
print(success)


image_request = airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)
responses = client.simGetImages([image_request],vehicle_name="DroneLeader")



#cv2.imwrite(str(val)+'_t.png', img_rgb)

response = responses[0]
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

img_rgb = img1d.reshape(response.height, response.width, 3)
img_rgb = np.flipud(img_rgb)
unique, counts = np.unique(np.concatenate(img_rgb), axis=0, return_counts=True)
print(counts)
print(unique)

idx=np.where(unique == [120, 0, 200])
print(idx)

tot=sum(counts)
val=counts[idx[0]]/tot
print(counts)
print(tot)
print(val)

#client.getMultirotorState()

import airsim
import os
from PIL import Image
import numpy as np
import cv2
import math
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -100, 10).join()
client.moveToPositionAsync(0, 0, -100, 10).join()
print(client.getMultirotorState().kinematics_estimated.position)

client.moveToPositionAsync(120, -120, -30, 5).join()
client.moveToPositionAsync(10, 0, -30, 5).join()

client.moveToPositionAsync(-200, -900, -70, 10).join()

image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
responses = client.simGetImages([image_request])
response = responses[0]
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(response.height, response.width, 3)
image = Image.fromarray(img_rgb)
im_final = np.array(image.resize((128, 128)))
img = np.ascontiguousarray(im_final, dtype=np.float32)
cv2.imwrite('_u.png', img)

pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
pi = math.pi
change = pi
yaw = (yaw + change)
vx = math.cos(yaw);
vy = math.sin(yaw);
client.hoverAsync().join()
client.rotateToYawAsync(0).join
client.moveByVelocityZAsync(vx, vy, -30, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()


client.rotateToYawAsync(90).join