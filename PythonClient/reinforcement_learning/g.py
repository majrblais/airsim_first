def get_screen(drone_name):
    img,img_seg = env._get_obs()

    if drone_name == "DroneFollower1":
        img_tosend=img.copy()
        mask_=img_seg.copy()
        #find dron1 and fire1
        mask_[np.where((mask_ == [199, 26, 29]).all(axis=2))] = [10,10,10]
        mask_[np.where((mask_ == [70, 52, 146]).all(axis=2))] = [20,20,20]
        mask_[np.where((mask_ == [123, 21, 124]).all(axis=2))] = [30,30,30]
        #keep only 1 channel with background, d1 and f1
        mask1=mask_[:,:,0]
        a=mask1!=0
        b=mask1!=10 #d1
        c=mask1!=20 #d2
        d=mask1!=30
        #make sure there is no other values with np.where.IF there is conflicting values (e.i other than 0,10 or 30) then switch it to 0. should not happen
        t=a|b|c|d
        newm=np.where(t,mask1,0)
        mask=torch.from_numpy(newm) 
        #find ids, remove first (background)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        #get masks->then boxes
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)  
        
        #draw boxes, drone will ALWAYS be green while destination will always be red. This allows us to add infinite drones.
        #d1,d2,p1 (green,red,blue)
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[0][0]),int(boxes[0][1])), (int(boxes[0][2]),int(boxes[0][3])), (0, 0, 255) )
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[1][0]),int(boxes[1][1])), (int(boxes[1][2]),int(boxes[1][3])), (0, 255, 0) )
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (255, 0, 0) )
        
        #img_tosend=cv2.resize(img_tosend,(512,512))
        cv2.imwrite('p1.png',img_tosend)
        #transpose, resize and transform into tensor
        img = img_tosend.transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32) #/ 255
        img = torch.from_numpy(img)
        
        return resize(img).unsqueeze(0)
        
    elif drone_name == "DroneFollower2":
        img_tosend=img.copy()
        mask_=img_seg.copy()
        mask_[np.where((mask_ == [199, 26, 29]).all(axis=2))] = [10,10,10]
        mask_[np.where((mask_ == [70, 52, 146]).all(axis=2))] = [20,20,20]
        mask_[np.where((mask_ == [214, 254, 86]).all(axis=2))] = [40,40,40]
        mask1=mask_[:,:,0]
        a=mask1!=0
        b=mask1!=10 #d1
        c=mask1!=20 #d2
        d=mask1!=40
        t=a|b|c|d
        newm=np.where(t,mask1,0)
        mask=torch.from_numpy(newm) 
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        
        #d1,d2,p2 (red,green,blue)
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[0][0]),int(boxes[0][1])), (int(boxes[0][2]),int(boxes[0][3])), (0, 255, 0) )
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[1][0]),int(boxes[1][1])), (int(boxes[1][2]),int(boxes[1][3])), (0, 0, 255) )
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (255, 0, 0) )

        cv2.imwrite('p2.png',img_tosend)
        img = img_tosend.transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32) #/ 255
        img = torch.from_numpy(img)
        return resize(img).unsqueeze(0)
        
    else:
        print("crititcal error, F")
        exit()
