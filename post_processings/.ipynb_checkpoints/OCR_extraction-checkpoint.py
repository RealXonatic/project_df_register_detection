import torch
import matplotlib.pyplot as plt
import pytesseract
import cv2 as cv
import numpy as np


def apply_erosion(tmp_img):

    if tmp_img.shape[0]==0 or tmp_img.shape[1]==0:
        return  tmp_img
        
    tmp_img_prec=tmp_img.copy()
    tmp_img[tmp_img_prec==255]=0
    tmp_img[tmp_img_prec==0]=255

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    tmp_img_final = cv.erode(tmp_img,kernel,iterations = 1).copy()

    tmp_img[tmp_img_final==255]=0
    tmp_img[tmp_img_final==0]=255

    return tmp_img

    
def extract_words_bb(box,img,show=False,erosion=False):

    if img.shape[0]==0 or img.shape[1]==0:
        return ""
    
    
    min_y=int(min(box[3].item(),box[1].item()))
    max_y=int(max(box[3].item(),box[1].item()))

    min_x=int(min(box[0].item(),box[2].item()))
    max_x=int(max(box[0].item(),box[2].item()))

    if max_x-min_x==0 or max_y-min_y==0:
        return ""
    
    tmp_img_prec=img[min_y:max_y,min_x:max_x]

    tmp_img=tmp_img_prec.copy()

    if tmp_img.shape[0]==0 or tmp_img.shape[1]==0:
        return ""
    
   
    if erosion:
        tmp_img=apply_erosion(tmp_img)

        

    
    
    
        
    txt=pytesseract.image_to_string(tmp_img, lang='eng')
        
    if show:
        plt.figure()
        plt.title(txt[:-1])
        plt.imshow(tmp_img,cmap="grey")
        plt.show()
        
    return txt


def extract_bb(box,img,show=False,erosion=False,word=None):
    

    min_y=int(min(box[3].item(),box[1].item()))
    max_y=int(max(box[3].item(),box[1].item()))

    min_x=int(min(box[0].item(),box[2].item()))
    max_x=int(max(box[0].item(),box[2].item()))
    
    tmp_img_prec=img[min_y:max_y,min_x:max_x]

    tmp_img=tmp_img_prec.copy()
    
    if erosion:

        tmp_img=apply_erosion(tmp_img)
        
    if show:
        plt.figure()
        plt.title("erosion")
        plt.imshow(tmp_img,cmap="grey")
        plt.show()


   
    tmp_L_bb=pytesseract.image_to_boxes(tmp_img)
    tmp_L_bb=tmp_L_bb.split("\n")
    L_bb=[]
    L_letters=[]

    idx_start=None

    

    for k in tmp_L_bb:
        splitted_bb=k.split(" ")
        if len(splitted_bb)>3:
            L_letters.append(splitted_bb[0])
            L_bb.append(splitted_bb[1:])


    if word!=None:
        for tmp_idx_word in range(0,len(L_bb)-len(word)):
            bool_verif=True
            for tmp_verif in range(0,len(word)):
                if L_letters[tmp_idx_word+tmp_verif].lower()!=word[tmp_verif]:
                    bool_verif=False
                    break
            if bool_verif:
                idx_start=tmp_idx_word
                break

    if word==None:
        idx_start=0
    else:
        if idx_start==None:
            return None,None,None,None

    if len(L_bb)==0:
        return None,None,None,None
                
    

    x_min=int(L_bb[idx_start][1])
    y_min=int(L_bb[idx_start][0])
    x_max=int(L_bb[idx_start][3])
    y_max=int(L_bb[idx_start][2])

    for tmp_b in range(idx_start,len(L_bb)):
        
        x_min=min(x_min,int(L_bb[tmp_b][1]))
        y_min=min(y_min,int(L_bb[tmp_b][0]))
        x_max=max(x_max,int(L_bb[tmp_b][1])+int(L_bb[tmp_b][3]))
        y_max=max(y_max,int(L_bb[tmp_b][0]))

    if word!=None and (word.lower()=="fig." or word.lower()=="table" ):
        x_max=min(int(L_bb[idx_start][3]),int(L_bb[tmp_b][3]))

    return x_min,x_max,y_min,y_max




def extract_bb_reforged(box,img,show=False,erosion=False,word=None,tol=75):
    

    min_y=int(min(box[3].item(),box[1].item()))
    max_y=int(max(box[3].item(),box[1].item()))

    min_x=int(min(box[0].item(),box[2].item()))
    max_x=int(max(box[0].item(),box[2].item()))
    
    tmp_img_prec=img[min_y:max_y,min_x:max_x]

    tmp_img=tmp_img_prec.copy()
    if tmp_img.shape[0]==0 or tmp_img.shape[1]==0 :
        return None,None,None,None
    if erosion:

        tmp_img=apply_erosion(tmp_img)
        
    if show:
        plt.figure()
        plt.title("erosion")
        plt.imshow(tmp_img,cmap="grey")
        plt.show()


    
    tmp_L_bb=pytesseract.pytesseract.image_to_data(tmp_img)
    
    tmp_L_bb=tmp_L_bb.split("\n")
    
    L_letters=[]
    L_bb=[]

    idx_start=None

    
   
    for k in tmp_L_bb[1:]:
        
        splitted_bb=k.split("\t")

        if len(splitted_bb)<3 or int(splitted_bb[10])==-1:
            continue
        
        L_letters.append(splitted_bb[11])
        L_bb.append(splitted_bb[6:10])

    
    if word!=None:
        for tmp_idx_word in range(0,len(L_letters)):
            bool_verif=True
            if L_letters[tmp_idx_word].lower().replace(" ","")==word.lower():
                idx_start=tmp_idx_word
                break

    
    if word==None:
        idx_start=0
    else:
        if idx_start==None:
            return None,None,None,None

    
    if len(L_bb)==0:
        return None,None,None,None
                
    
    idx_start=max(0,idx_start)
    x_min=int(L_bb[idx_start][0])
    x_word=int(L_bb[idx_start][0])
    y_min=int(L_bb[idx_start][1])
    x_max=int(L_bb[idx_start][2])+int(L_bb[idx_start][0])
    y_max=int(L_bb[idx_start][3])+int(L_bb[idx_start][1])

    for tmp_b in range(len(L_bb)):
        if int(L_bb[tmp_b][1])<int(L_bb[idx_start][1])+10:
            continue
        if x_word<int(L_bb[tmp_b][0]) and np.sqrt((int(L_bb[idx_start][1])-int(L_bb[tmp_b][1]))**2)<tol:
            
        
            x_min=min(x_min,int(L_bb[tmp_b][0]))
            y_min=min(y_min,int(L_bb[tmp_b][1]))
            x_max=max(x_max,int(L_bb[tmp_b][2])+int(L_bb[tmp_b][0]))
            y_max=max(y_max,int(L_bb[tmp_b][3])+int(L_bb[tmp_b][1]))


    
    
   
    return y_min,y_max,x_min,x_max

