import airsim
import os
from PIL import Image
import numpy as np
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
#print(client.getMultirotorState())

client.takeoffAsync().join()
client.moveToPositionAsync(-10, 10, -10, 5).join()
success = client.simSetSegmentationObjectID("ob", 54);
print(success)


image_request = airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)
responses = client.simGetImages([image_request])

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