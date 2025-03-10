import torch
import matplotlib.pyplot as plt
import pytesseract
import cv2 as cv2
import numpy as np

from .OCR_extraction import *
from .box_utils import *
from .utils_VISHAY import *
from .utils_STMICRO import *
from .boxes import *


def  overlapping_pictures_tables(boxes,scores,class_labels,bools,page_id):
    for id_box in range(boxes.shape[0]):
        if not class_labels[id_box]==0: 
            continue
            
        for idx_box in range(boxes.shape[0]):
            if not class_labels[idx_box] ==7: 
                continue
            cond_d = ratio_a_in_b(boxes[idx_box],boxes[id_box])>0.5

            if page_id==1:
                if cond_d and scores[id_box]+15>scores[idx_box]:
                    bools[idx_box]=0
            
            else:
                if cond_d and scores[id_box]>scores[idx_box]:
                    bools[idx_box]=0
    return bools

def  overlapping_sechead_text(boxes,scores,class_labels,bools):
    #text 1
    #sec head 6
    for id_box in range(boxes.shape[0]):
        if not class_labels[id_box]==6: 
            continue
            
        for idx_box in range(boxes.shape[0]):
            if not class_labels[idx_box]==1: 
                continue
            cond_d = ratio_a_in_b(boxes[id_box],boxes[idx_box])>0.7
            
            #couper le text en 2 élements / appel recursif sur la fun jusqu'a plus de changement
    
    return bools

def  overlapping_sechead_pagehead(boxes,scores,class_labels,bools):
    for id_box in range(boxes.shape[0]):
        if not class_labels[id_box]==6: 
            continue
            
        for idx_box in range(boxes.shape[0]):
            if not class_labels[idx_box]==4: 
                continue
            cond_d = ratio_a_in_b(boxes[idx_box],boxes[id_box])>0.5 
           
            if cond_d :
                
                if scores[idx_box]<0.9:
                    bools[idx_box]=0
                    
    return bools


def correct_false_sechead(boxes,scores,class_labels,img):

    for id_box in range(boxes.shape[0]):
        if class_labels[id_box]==6:
            words=extract_words_bb(boxes[id_box],img,show=False,erosion=False)
            if "." in words[-3:]:
                class_labels[id_box]=1
    return class_labels
       


def supress_overlapping_boxes_light(id_box,boxes,scores,class_labels,booleans,page_id,img):
    #this function supress text elements nested inside box 
    
    #{'Table': 0, 'Text': 1, 'Title': 2, 'Caption':3 , 'Page-header': 4,'Page-footer': 5, 'Section-header': 6, 'Picture': 7}
    box=boxes[id_box]

    if booleans[id_box]==0:
        return False,boxes,booleans,class_labels

    if class_labels[id_box].item() in [0,7]:

        
        for idx_box in range(boxes.shape[0]):
            tmp_box=boxes[idx_box]

            if id_box==idx_box:
                continue

            if class_labels[idx_box]==class_labels[id_box] and ratio_a_in_b(tmp_box,box)>0.90:
                
                boxes[id_box,0]=min(boxes[idx_box,0],boxes[id_box,0])
                boxes[id_box,1]=max(boxes[idx_box,1],boxes[id_box,1])
                boxes[id_box,2]=max(boxes[idx_box,2],boxes[id_box,2])
                boxes[id_box,3]=min(boxes[idx_box,3],boxes[id_box,3])
    
                
                    
                booleans[idx_box]=0.0
                verif=True

                return True,boxes,booleans ,class_labels
            
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
        cond_b = ratio_a_in_b(tmp_box,box)>0.01 and class_labels[id_box]==1 and class_labels[idx_box]==1 and dist_xmax<50

        

        #cond_3 both class are text and ...

        #idx is above id
        dist_ymax=np.sqrt((boxes[idx_box,1].item()-boxes[id_box,3].item())**2)

        #xmin idx < xmax id
        #xmin idx > xmin_idx-30

        #xmax idx -30 <= xmax id
        #xmin idx +30  >= xmin id
        dist_xmax= ((boxes[idx_box,0]<= boxes[id_box,2]) and (boxes[idx_box,0]>= boxes[id_box,0]-30)) or ((boxes[idx_box,2]-30<= boxes[id_box,2]) and (boxes[idx_box,0]+30>= boxes[id_box,0]))

        

        dist_xmax= dist_xmax or ((boxes[id_box,0]<= boxes[idx_box,2]) and (boxes[id_box,0]>= boxes[idx_box,0]-30)) or ((boxes[id_box,2]-30<= boxes[idx_box,2]) and (boxes[id_box,0]+30>= boxes[idx_box,0]))


        
    
        
        cond_c = (class_labels[id_box] in [1]) and (class_labels[idx_box]in [1]) and dist_ymax<100 and dist_xmax

        

        #cond_4
        cond_d = ratio_a_in_b(tmp_box,box)>0.5 and class_labels[id_box] in [0,7] and class_labels[idx_box] in [0,7] and scores[id_box]>scores[idx_box]
       
        
        dist_ymax=min(dist_ymax,np.sqrt((boxes[idx_box,3].item()-boxes[id_box,1].item())**2))

        

        #caption nested with sec_head
        if ratio_a_in_b(tmp_box,box)>0.30 and class_labels[id_box]==6 and class_labels[idx_box]==3 and scores[id_box]>0.95:

            boxes[id_box,0]=min(boxes[idx_box,0],boxes[id_box,0])
            boxes[id_box,1]=max(boxes[idx_box,1],boxes[id_box,1])
            boxes[id_box,2]=max(boxes[idx_box,2],boxes[id_box,2])
            boxes[id_box,3]=min(boxes[idx_box,3],boxes[id_box,3])

            booleans[idx_box]=False
            verif=True
            return verif,boxes,booleans,class_labels
            

        

        
        
        if cond_a or cond_b  or cond_c or cond_d   :
            
            
                
            box_verif=torch.zeros(4)
            box_verif[0]=int(min(boxes[idx_box,0],boxes[id_box,0]))
            box_verif[1]=int(max(boxes[idx_box,1],boxes[id_box,1]))
            box_verif[2]=int(max(boxes[idx_box,2],boxes[id_box,2]))
            box_verif[3]=int(min(boxes[idx_box,3],boxes[id_box,3]))

            tmp_w=int(box_verif[0].item())
            tmp_h=int(box_verif[3].item())

            if not  ((class_labels[id_box] in [1]) and (class_labels[idx_box]in [1])):
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

            
            break
            
    return verif,boxes,booleans,class_labels



def correct_bbox_all_except_pictures_tables(f_b,f_s,f_l,img_BW):
    for idx_box in range(f_b.shape[0]):
        if not (f_l[idx_box].to(int).item()) in [0,7]:

            box_prec=f_b[idx_box].clone()
            box_prec_area=compute_area_of_box(box_prec)
            f_b=recalage_boxes_xgauche(idx_box,f_b,img_BW,step=15,attempt=True)
            f_b=recalage_boxes_xdroit(idx_box,f_b,img_BW,step=15,attempt=True)
            f_b=recalage_boxes_yhaut(idx_box,f_b,img_BW,step=2,attempt=True)
            f_b=recalage_boxes_ybas(idx_box,f_b,img_BW,step=2,attempt=True)

            f_b=supress_blank_area(idx_box,f_b,img_BW)

            box_new_area=compute_area_of_box(f_b[idx_box])
            
            if (box_new_area/box_prec_area)>2.0:
                f_b[idx_box]=box_prec
    return f_b,f_s,f_l


def correct_bbox_pictures_tables(f_b,f_s,f_l,img_BW):
    tmp_img_only_PT=img_BW.copy()
    for idx_box in range(f_b.shape[0]):
        if not (f_l[idx_box].to(int).item()) in [0,7]:

            xmin=int(f_b[idx_box,0].item())
            ymax=int(f_b[idx_box,1].item())
            xmax=int(f_b[idx_box,2].item())
            ymin=int(f_b[idx_box,3].item())
    
            tmp_img_only_PT[ymin:ymax,xmin:xmax]=255

    for idx_box in range(f_b.shape[0]):
        if (f_l[idx_box].to(int).item()) in [0,7]:

            box_prec=f_b[idx_box].clone()
            box_prec_area=compute_area_of_box(box_prec)
            f_b=recalage_boxes_xgauche(idx_box,f_b,tmp_img_only_PT,step=5,attempt=True)
            f_b=recalage_boxes_xdroit(idx_box,f_b,tmp_img_only_PT,step=5,attempt=True)
            f_b=recalage_boxes_yhaut(idx_box,f_b,tmp_img_only_PT,step=5,attempt=True)
            f_b=recalage_boxes_ybas(idx_box,f_b,tmp_img_only_PT,step=5,attempt=True)


            f_b=supress_blank_area(idx_box,f_b,img_BW)

            box_new_area=compute_area_of_box(f_b[idx_box])
            
            if (box_new_area/box_prec_area)>1.3 or (box_new_area/box_prec_area)<0.95 :
                f_b[idx_box]=box_prec
    
    return f_b,f_s,f_l


def correct_overlapping_sec_head_over_text(boxes,scores,class_labels,page_id,img):

    

    while(True):
        verif_main_loop=False
        for id_box in range(boxes.shape[0]):
            if not (class_labels[id_box].to(int).item())==1:
                continue
        
            for idx_box in range(boxes.shape[0]):
                if not (class_labels[idx_box].to(int).item())==6:
                    continue
                    
                box_text=boxes[id_box]
                boc_sec_head=boxes[idx_box]
        
                if ratio_a_in_b(boc_sec_head,box_text)>0.01:
        
                    tmp_img_blank=img.copy()
        
                    for idx_tmp_box in range(boxes.shape[0]):
                        if idx_tmp_box!=id_box:
                            tmp_box_blank=boxes[idx_tmp_box]
                            xmin=int(tmp_box_blank[0].item())
                            ymax=int(tmp_box_blank[1].item())
                            xmax=int(tmp_box_blank[2].item())
                            ymin=int(tmp_box_blank[3].item())
        
                            tmp_img_blank[ymin-10:ymax+10,xmin-30:xmax+30]=255


                    
        
                    verif=True
                    while(verif):
                        verif,boxes,scores,class_labels=divide_blank_space_horizontal(id_box,boxes,scores,class_labels,tmp_img_blank,min_width=50,mw_nb=10)
                        if verif:
                            verif_main_loop=True
                            boxes=supress_blank_area(id_box,boxes,tmp_img_blank)
                            boxes=supress_blank_area(boxes.shape[0]-1,boxes,tmp_img_blank)
        if verif_main_loop:
            continue
        else:
            break



    for id_box in range(boxes.shape[0]):
            if not (class_labels[id_box].to(int).item())==1:
                continue
        
            for idx_box in range(boxes.shape[0]):
                if not (class_labels[idx_box].to(int).item())==6:
                    continue
                    
                box_text=boxes[id_box]
                box_sec_head=boxes[idx_box]

                if ratio_a_in_b(box_sec_head,box_text)>0.01:

                    if box_text[3]<box_sec_head[1]:
                        boxes[id_box,3]=box_sec_head[1]+15
                    


    
                
                
       
                
            
    return boxes,scores,class_labels



            
            
    

def traitement_boxes_lw(boxes,scores,class_labels,page_id,img):

    img_BW=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_BW=np.array(img_BW.copy())
    img_BW[img_BW<200]=0
    img_BW[img_BW>=200]=255
    
    
    f_b,f_s,f_l=supress_low_confidence_boxes(boxes,scores,class_labels,treshold=0.3)


    #for idx_box in range(f_b.shape[0]):
    #    f_l=correct_boxes_with_content(idx_box,f_b,f_s,f_l,img,bottom_line,header_line)


 

    f_l=correct_false_sechead(f_b,f_s,f_l,img_BW)
    f_b,f_s,f_l=box_fusion(f_b,f_s,f_l,page_id,img,supress_overlapping_boxes_light)

    f_b,f_s,f_l=correct_overlapping_sec_head_over_text(f_b,f_s,f_l,page_id,img_BW)


    tmp_bools=torch.ones(f_b.shape[0])
    tmp_bools=overlapping_pictures_tables(f_b,f_s,f_l,tmp_bools,page_id)

    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    tmp_bools=torch.ones(f_b.shape[0])
    tmp_bools=overlapping_sechead_pagehead(f_b,f_s,f_l,tmp_bools)

    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    f_b,f_s,f_l=correct_bbox_all_except_pictures_tables(f_b,f_s,f_l,img_BW)
    f_b,f_s,f_l=correct_bbox_pictures_tables(f_b,f_s,f_l,img_BW)

    

    

    



    

    
    
    

    
    return True,f_b,f_s,f_l,(None,None,None,None)