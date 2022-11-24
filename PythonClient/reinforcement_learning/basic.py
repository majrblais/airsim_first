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
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -30, 5).join()
client.moveToPositionAsync(120, -120, -30, 5).join()


image_request = airsim.ImageRequest(3, airsim.ImageType.Scene, False, False)
responses = client.simGetImages([image_request])
response = responses[0]
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(response.height, response.width, 3)
image = Image.fromarray(img_rgb)
im_final = np.array(image.resize((128, 128)))
img = np.ascontiguousarray(im_final, dtype=np.float32)
cv2.imwrite('_u.png', img)