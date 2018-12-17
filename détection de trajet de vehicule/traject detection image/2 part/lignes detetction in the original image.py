
# coding: utf-8

# In[37]:


import cv2
import numpy as np
import matplotlib.pyplot as plt 

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
#pour afficher l'image qu'on va traiter decommente les 3 lignes suivantes
#cv2.imshow('original image', image)
#cv2.waitKey()
#cv2.destroyAllWindows()

def canny (image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)  #convertion de RGB -> gray
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # appliquer un filtre gaussian de taille 5*5 afin de miniser les bruits 
    canny = cv2.Canny(blur, 50, 150) #gradiant pour la detection de changement de luminosité
    return canny

def region_of_interet (image): #creation de la mask 
    height = image.shape[0]
    polygone = np.array([[(200, height), (1100, height), (550, 250)]]) # les coordonées de polygone 
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygone, 255) #Mask contain the polygone in white et le reste en noir 
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, mask
def afficher_lignes(images, lignes):
    ligne=np.zeros_like(image)
    if lignes is not None:
        for line in lignes:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(ligne, (x1, y1), (x2, y2), (0, 255, 0), 8)
    return ligne
canny=canny(lane_image)   
masked_image, mask = region_of_interet(canny)
lignes = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
ligne_image = afficher_lignes(canny, lignes)
ligne_original= cv2.addWeighted(ligne_image, 0.8, image, 1, 1)
#decommente la ligne pour afficher le resultat 
#plt.imshow(canny)  #on l'utilise pour identifier la zone d'interet dans l'image canny  
#plt.show()
#cv2.imshow('mask', mask)
#cv2.imshow('detection de bordure', canny)
#cv2.imshow('masked_image', masked_image)
#cv2.imshow('ligne_image', ligne_image)
cv2.imshow('ligne_original', ligne_original)
cv2.waitKey()
cv2.destroyAllWindows()

