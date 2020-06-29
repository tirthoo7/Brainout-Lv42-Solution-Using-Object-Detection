#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os


# In[2]:


def display(img):

    cv2.imshow('Display', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# In[3]:


def find_obj_fatch():
    
    image = cv2.imread('./objs.png')
    cv2.imshow('0 - Original Image', image)
    cv2.waitKey(0)

    # Create a black image with same dimensions as our loaded image(0->Black)
    blank_image = np.zeros((image.shape[0], image.shape[1], 3))

    # Create a copy of our original image
    orginal_image = image
    copy_orginal=image

    # Grayscale our image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 50, 200)
    cv2.imshow('1 - Canny Edges', edged)
    cv2.waitKey(0)

    # Find contours and print how many were found
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    def x_cord_contour(contours):
        #Returns the X cordinate for the contour centroid
        if cv2.contourArea(contours) > 10:
            M = cv2.moments(contours)
            return (int(M['m10']/M['m00']))
        else:
            pass


    def label_contour_center(image, c):
        # Places a red circle on the centers of contours
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Draw the countour number on the image
        cv2.circle(image,(cx,cy), 10, (0,0,255), -1)
        return image


    # Load our image
    image = cv2.imread('./objs.png')
    orginal_image = image.copy()


    # Computer Center of Mass or centroids and draw them on our image
    for (i, c) in enumerate(contours):
        orig = label_contour_center(image, c)

    cv2.imshow("Contour Centers ", image)
    cv2.waitKey(0)

    # Sort by left to right using our x_cord_contour function
    contours_left_to_right = sorted(contours, key = x_cord_contour, reverse = False)


    # Labeling Contours left to right
    for (i,c)  in enumerate(contours_left_to_right):
    #     copy_orignal=orignal_image.copy()
        cv2.drawContours(orginal_image, [c], -1, (0,0,255), 3)  
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    #     cv2.putText(orginal_image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        cv2.imshow('Left to Right Contour', orginal_image)
        cv2.waitKey(0)
        (x, y, w, h) = cv2.boundingRect(c)  
    #     return bounded rectangle

        # Let's now crop each contour and save these images
        cropped_contour = copy_orginal[y:y + h, x:x + w]
        image_name = "obj_number_" + str(i+1) + ".png"
        print (image_name)
        cv2.imwrite("./Objects_Identify/"+image_name, cropped_contour)
        cv2.imshow('Objects',cropped_contour)

    cv2.destroyAllWindows()



# In[ ]:





# In[4]:


def target_img_seg():
    image = cv2.imread('./target_image.png')
    cv2.imshow('0 - Original Image', image)
    cv2.waitKey(0)

    # Create a black image with same dimensions as our loaded image(0->Black)
    blank_image = np.zeros((image.shape[0], image.shape[1], 3))

    # Create a copy of our original image
    orginal_image = image

    # Grayscale our image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 50, 200)
    cv2.imshow('1 - Canny Edges', edged)
    cv2.waitKey(0)

    # Find contours and print how many were found
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print ("Number of contours found = ", len(contours))

    #Draw all contours
    cv2.drawContours(blank_image, contours, -1, (0,255,0), 1)
    cv2.imshow('2 - All Contours over blank image', blank_image)
    cv2.waitKey(0)

    # Draw all contours over blank image
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    cv2.imshow('3 - All Contours', image)
    cv2.waitKey(0)
    # contours_left_to_right = sorted(contours, key = x_cord_contour, reverse = False)


    # Labeling Contours left to right
    for (i,c)  in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)  
    #     return bounded rectangle

        # Let's now crop each contour and save these images
        cropped_contour = orginal_image[y:y + h, x:x + w]
        image_name = "target_obj_number_" + str(i+1) + ".png"
        print (image_name)
        cv2.imwrite("./target_img_seg/"+image_name, cropped_contour)
        cv2.imshow('Target Image Segmentation',cropped_contour)

    cv2.destroyAllWindows()


# In[ ]:





# In[5]:


def match():
    # Load input image and convert to grayscale
    image = cv2.imread('./target_image.png')
    display(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Template image
   
    
#     path='Objects_full_match/'
    path='Objects_Identify/'
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       
        
        template = cv2.imread(imagePath)
        cv2.imshow("Object To Found",template)
        cv2.waitKey(0)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        #Create Bounding Box
        top_left = max_loc
        bottom_right = (top_left[0] + 50, top_left[1] + 50)
        cv2.rectangle(image, top_left, bottom_right, (0,0,255), 3)
        display(image)

       
            
                              
                              
    


# In[6]:


match()


# In[8]:


target_img_seg()


# In[6]:


find_obj_fatch()


# In[ ]:




