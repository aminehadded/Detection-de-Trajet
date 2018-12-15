
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable


# image.shapereturns h, w, chan in a array
 
def canny (image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #convertion de RGB -> gray
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # appliquer un filtre gaussian de taille 5*5 afin de miniser les bruits 
    canny = cv2.Canny(blur, 50, 150) #gradiant pour la detection de changement de luminosité détection des contours
    return canny

def make_coordinates(image, line_parameters):
    try:
        some_object_iterator = iter(line_parameters)
        pente, intercept =line_parameters  #(pente, intercept)
        y1=image.shape[0]  # l'origine par rapport à l'hauteur tous les lignes commencent de la fin de l'image 
        y2=int ((3/5)*y1) # une autre point proportionnelle à y1
        # voir l'image "axis"  au niveau de dossier full_code 
        x1 = int((y1 - intercept)/pente) # determination de coordonné de point selon l'axe x qui correspond à y1 
        x2 = int((y2 - intercept)/pente) # determination de coordonné selon point l'axe x qui correspond à y2 
        return np.array([x1, y1, x2, y2])

    except TypeError as te:
        print (line_parameters, 'is not iterable')
       
def moyenne_pent_intercept(image, lines):
    gauche_forme = []
    droite_forme = []
    y1=image.shape[0] 
    y2=int ((3/5)*y1)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters =  np.polyfit((x1, x2), (y1, y2), 1) # determination des parameters pente et intercept par la modelisation de la relation entre x et y par une équation de 1ere degré/
        pente =parameters[0]
        intercept = parameters[1]
        if (pente<0):
            gauche_forme.append((pente, intercept))  # au niveau de lignes de gauches y aygmente et x diminue donc pente negative
        else:
            droite_forme.append((pente, intercept)) #l'inverse pour droite y augmente x augmente
    gauche_forme_moyen = np.average(gauche_forme, axis=0) # la moyenne des parameters afin de tracer une seule ligne à gauche
    droite_forme_moyen = np.average(droite_forme, axis=0) # la moyenne des parameters afin de tracer une seule ligne à droite
    gauche_ligne = make_coordinates(image, gauche_forme_moyen) #derteminer les coordonnées de lgne à gauche
    droite_ligne = make_coordinates(image, droite_forme_moyen) #detereminer les coordonnées de ligne à droite
    return np.array([gauche_ligne, droite_ligne]) # return 2D array contient les coordonnées de deux lignes 


def region_of_interet (image): #creation de la mask 
    height = image.shape[0]
    polygone = np.array([[(200, height), (1100, height), (550, 250)]]) # les coordonées de polygone 
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygone, 255) #Mask contain the polygone in white et le reste en noir 
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def afficher_lignes(images, lignes):
    ligne=np.zeros_like(images)
    if lignes is not None:
        for line in lignes:
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(ligne, (x1, y1), (x2, y2), (0, 255, 0), 8)
    return ligne
# canny=canny(lane_image)   
# masked_image, mask = region_of_interet(canny)
# lignes = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# lignes_moyennes = moyenne_pent_intercept(masked_image, lignes)
# ligne_image = afficher_lignes(canny, lignes_moyennes)
# ligne_original= cv2.addWeighted(ligne_image, 0.8, image, 1, 1) 
# #decommente la ligne pour afficher le resultat 
# #plt.imshow(canny)  #on l'utilise pour identifier la zone d'interet dans l'image canny  
# #plt.show()
# #cv2.imshow('mask', mask)
# #cv2.imshow('detection de bordure', canny)
# #cv2.imshow('masked_image', masked_image)
# #cv2.imshow('ligne_image', ligne_image)
# #cv2.imshow('lignes_moyennes', ligne_image)
# cv2.imshow('ligne_original', ligne_original)
# cv2.waitKey()
# cv2.destroyAllWindows()

# mettre le path de video test2 
cap = cv2.VideoCapture(r'C:\Users\mohamed\Desktop\computer vsison\détection de tajet de vehicule\test2.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read() # capture each frame from the video 
    if ret==True:
        canny_image = canny(frame)
        cropped_image = region_of_interet(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        average_lines = moyenne_pent_intercept(frame, lines)
        line_image = afficher_lignes(frame, average_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) 
        out.write(frame)
        cv2.imshow('result', combo_image)  
        if cv2.waitKey(1) == ord('q'):  # attend 1 millisecond entre frames, ord() verfié si le button q est pressé 
            break

cap.release()  # stop video
out.release()
cv2.destroyAllWindows()  # fermer window

