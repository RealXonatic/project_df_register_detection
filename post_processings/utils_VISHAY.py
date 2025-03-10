import torch
import matplotlib.pyplot as plt
import pytesseract
import cv2 as cv
import numpy as np

from .OCR_extraction import *
from .boxes import *  
from .box_utils import *

def find_captions_table_VISHAY(id_box,boxes,scores,class_labels,booleans,img):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y
   
        
    box=boxes[id_box]


    #for idx_box in range(boxes.shape[0]):
    #    tmp_box=boxes[idx_box]
    #    if class_labels[idx_box]==0 and ratio_a_in_b(tmp_box,box)>0.98:
    #        return boxes,scores,class_labels,booleans

    #find y level box
    tmp_y_haut=int(box[3].item())
    tmp_y_bas=int(box[3].item())
    tmp_x_droit=int(box[2].item()-50)
    tmp_x_gauche=int(box[0].item()+50)
    

    cpt_line=0
    cpt_interline=0
    while cpt_line<=2 and tmp_y_bas<img.shape[0]:
        if not (255 in img[tmp_y_bas,tmp_x_gauche:tmp_x_droit]) and cpt_interline==0:
            cpt_line+=1
            cpt_interline=1
        else:
            cpt_interline=0
            
        tmp_y_bas+=1

    
    
    tmp_newbox=torch.zeros((1,4))
    tmp_newbox[0,0]=int(box[0].item())
    tmp_newbox[0,1]=tmp_y_bas
    tmp_newbox[0,2]=int(box[2].item())
    tmp_newbox[0,3]=tmp_y_haut


    y_min,y_max,x_min,x_max=extract_bb(tmp_newbox[0],img,show=False,erosion=False)

    if y_min==None:
        y_min,y_max,x_min,x_max=extract_bb(tmp_newbox[0],img,show=False,erosion=True)
    else:
        y_min_eros,y_max_eros,x_min_eros,x_max_eros=extract_bb(tmp_newbox[0],img,show=False,erosion=True)
        if y_min_eros!=None:
            y_min=min(y_min,y_min_eros)
            x_min=min(x_min,x_min_eros)
            y_max=min(y_max,y_max_eros)
            x_max=min(x_max,x_max_eros)
            

    

    if y_min==None:
        return boxes,scores,class_labels,booleans
    

    mid_x=(x_min+x_max)/2.0
    mid_y=(y_min+y_max)/2.0
    width=(y_max-y_min)/4.0
    width_x=(x_max-x_min)/4.0

    tmp_newbox[0,0]=mid_x-width_x+int(box[0].item())
    tmp_newbox[0,1]=mid_y+width+int(box[3].item())
    tmp_newbox[0,2]=mid_x+width_x+int(box[0].item())
    tmp_newbox[0,3]=mid_y-width+int(box[3].item())

    
            



    
        

    

    boxes=torch.cat((boxes,tmp_newbox),dim=0)

    
    boxes=recalage_boxes_yhaut(boxes.shape[0]-1,boxes,img,step=2)
    boxes=recalage_boxes_ybas(boxes.shape[0]-1,boxes,img,step=2)
    boxes=recalage_boxes_xgauche(boxes.shape[0]-1,boxes,img)
    boxes=recalage_boxes_xdroit(boxes.shape[0]-1,boxes,img)
    
    

    

    boxes[boxes.shape[0]-1,0]=max(boxes[boxes.shape[0]-1,0],232)
    boxes=recalage_boxes_xgauche(boxes.shape[0]-1,boxes,img,step=1)

    


    boxes=supress_blank_area(boxes.shape[0]-1,boxes,img)
    
    
    if np.sqrt((boxes[boxes.shape[0]-1,1].item()-boxes[boxes.shape[0]-1,3].item())**2)>250:
        
    
        
        
    
        return boxes[:-1],scores,class_labels,booleans
        
    tmp_score=torch.empty(1)
    tmp_label=torch.empty(1)
    tmp_bool=torch.empty(1)
    
    tmp_score[0]=1.5
    tmp_label[0]=0
    tmp_bool[0]=1
    
    scores=torch.cat((scores,tmp_score),dim=0)
    class_labels=torch.cat((class_labels,tmp_label),dim=0)
    booleans=torch.cat((booleans,tmp_bool),dim=0)

    
    

    return boxes,scores,class_labels,booleans



        


def correct_wrong_labels_VISHAY(id_box,boxes,scores,class_labels,booleans,img,page_id):
    box=boxes[id_box]

    cpt_overlaps=0

    
    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

        
        if ratio_a_in_b(tmp_box,box)>0.90 and class_labels[idx_box]==7 and class_labels[id_box]==8:
            class_labels[idx_box]=0

        if ratio_a_in_b(tmp_box,box)>0.99 and class_labels[idx_box]==9 and class_labels[id_box]==8:
            class_labels[idx_box]=0

    if class_labels[id_box]==7 and extract_words_bb(box,img).lower()=="note":
        class_labels[id_box]=9

    if class_labels[id_box]==10 and box[1].item()>=2000:
        class_labels[id_box]=5

    if (class_labels[id_box] in [5,7]) and box[3]<=600 and box[1]>=250 :
        mid_page_x=img.shape[1]/2.0
        mid_box_x=(box[0]+box[2])/2.0
        box=boxes[id_box]
        w=np.sqrt((box[0].item()-box[2].item())**2)

        if np.sqrt((mid_page_x-mid_box_x)**2)<100 and w>100:
            class_labels[id_box]=10
            
        

    if class_labels[id_box]==9 and box[3].item()>=3000:
        class_labels[id_box]=4

    if class_labels[id_box]==7 and extract_words_bb(box,img).lower()[:4]=="www." and box[1].item()<=500:
        class_labels[id_box]=5

    if class_labels[id_box]==6:
        tmp_txt=extract_words_bb(box,img,show=False,erosion=True)

        if "information" in tmp_txt.lower():
            class_labels[id_box]=8
            
            


    
        
            

    
        

    return class_labels,booleans


def get_line_footer_VISHAY(img,y_start):
    y_res=None
    L_res=[]

    cpt_black_lines=0

    for tmp_y in range(y_start,img.shape[0]):
        if img[tmp_y,500:2000].mean()==0:
            y_res=tmp_y
            L_res.append(tmp_y)
    return L_res

def get_line_header_VISHAY(img,y_end):
    y_res=None

    cpt_black_lines=0
    L_res=[]

    for tmp_y in range(20,y_end):
        if img[tmp_y,250:1750].mean()==0 or img[tmp_y,600:2100].mean()==0 :
            y_res=tmp_y

            
            if len(L_res)>=1:
                if np.sqrt((tmp_y-L_res[0])**2)<15:
                    L_res.append(tmp_y)
            else:
                L_res.append(tmp_y)
            
    return L_res



def supress_overlapping_boxes(id_box,boxes,scores,class_labels,booleans,page_id,img):
    #this function supress text elements nested inside box 
    

    box=boxes[id_box]

    if booleans[id_box]==0:
        return False,boxes,booleans,class_labels

    if class_labels[id_box].item() in [6,8]:
        return False,boxes,booleans ,class_labels

    verif=False
    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

            
            


        #cond_1 - same class and overlapping
        cond_a = ratio_a_in_b(tmp_box,box)>0.40 and class_labels[id_box]==class_labels[idx_box]

        dist_xmax=np.sqrt((boxes[idx_box,0].item()-boxes[id_box,0].item())**2)

        #cond_2 both class are text, overlap and have similar widths
        cond_b = ratio_a_in_b(tmp_box,box)>0.01 and class_labels[id_box]==9 and class_labels[idx_box]==9 and dist_xmax<50

        #cond_3 both class are text and ...
        dist_ymax=np.sqrt((boxes[idx_box,1].item()-boxes[id_box,3].item())**2)
        dist_xmax= ((boxes[idx_box,0]<= boxes[id_box,2]) and (boxes[idx_box,0]>= boxes[id_box,0])) or ((boxes[idx_box,2]<= boxes[id_box,2]) and (boxes[idx_box,0]>= boxes[id_box,0]))
        dist_xmax= dist_xmax or (((boxes[id_box,0]<= boxes[idx_box,2]) and (boxes[id_box,0]>= boxes[idx_box,0])) or ((boxes[id_box,2]<= boxes[idx_box,2]) and (boxes[id_box,0]>= boxes[idx_box,0])))
        
        cond_c = (class_labels[id_box] in [0,9]) and (class_labels[idx_box]in [0,9]) and dist_ymax<30 and dist_xmax

        
        

        #cond_4
        cond_d=None
        dist_ymax=min(dist_ymax,np.sqrt((boxes[idx_box,3].item()-boxes[id_box,1].item())**2))
        if (not class_labels[id_box] in [6,8,0,4,5]) and (class_labels[idx_box]==6):
            tmp_txt=extract_words_bb(box,img,show=False,erosion=False)
            
            cond_d=("view" in tmp_txt) and len(tmp_txt)<=30 and dist_ymax<150 and (not boxes[id_box,2]<boxes[idx_box,0]) 
        
        

        

        
        if cond_a or cond_b  or cond_c or cond_d  :
            if class_labels[id_box]==0 and scores[id_box]==1.50:
                booleans[idx_box]=False
                verif=True
                return verif,boxes,booleans,class_labels

            if class_labels[idx_box]==0 and scores[idx_box]==1.50:
                booleans[id_box]=False
                verif=True
                return verif,boxes,booleans,class_labels

            
                
            box_verif=torch.zeros(4)
            box_verif[0]=int(min(boxes[idx_box,0],boxes[id_box,0]))
            box_verif[1]=int(max(boxes[idx_box,1],boxes[id_box,1]))
            box_verif[2]=int(max(boxes[idx_box,2],boxes[id_box,2]))
            box_verif[3]=int(min(boxes[idx_box,3],boxes[id_box,3]))

            tmp_w=int(box_verif[0].item())
            tmp_h=int(box_verif[3].item())

            tmp_img=img.copy()
            tmp_img[int(boxes[idx_box,3].item()):int(boxes[idx_box,1].item()),int(boxes[idx_box,0].item()):int(boxes[idx_box,2].item())]=255
            tmp_img[int(boxes[id_box,3].item()):int(boxes[id_box,1].item()),int(boxes[id_box,0].item()):int(boxes[id_box,2].item())]=255
            
            img_verif=tmp_img[int(box_verif[3].item()):int(box_verif[1].item()),int(box_verif[0].item()):int(box_verif[2].item())].copy()
            

            test_cond= (0 in img_verif) and (255 in img_verif) 
            if test_cond:
                verif=False
                continue
            
            boxes[id_box,0]=min(boxes[idx_box,0],boxes[id_box,0])
            boxes[id_box,1]=max(boxes[idx_box,1],boxes[id_box,1])
            boxes[id_box,2]=max(boxes[idx_box,2],boxes[id_box,2])
            boxes[id_box,3]=min(boxes[idx_box,3],boxes[id_box,3])
            booleans[idx_box]=0.0
            verif=True

            if cond_d:
               
                class_labels[id_box]=6

                boxes=recalage_boxes_xgauche(id_box,boxes,img,step=20)
                boxes=recalage_boxes_xdroit(id_box,boxes,img,step=20)
                boxes=recalage_boxes_yhaut(id_box,boxes,img,step=20)
                boxes=recalage_boxes_ybas(id_box,boxes,img,step=20)

            
            break
            
    return verif,boxes,booleans,class_labels
            
            


    