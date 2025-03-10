import torch
import matplotlib.pyplot as plt
import pytesseract
import cv2 as cv
import numpy as np

import cv2
from .OCR_extraction import *
from .box_utils import *
from .utils_VISHAY import *
from .utils_STMICRO import *


def supress_low_confidence_boxes(boxes,scores,class_labels,treshold=0.15):
    final_boxes=torch.empty((0,4))
    final_scores=torch.empty((0))
    final_labels=torch.empty((0))

    for idx_box in range(boxes.shape[0]):
        tmp_box=boxes[idx_box]
    
        
        
        if scores[idx_box]<treshold :
            continue
        else:
            final_boxes=torch.cat((final_boxes,torch.unsqueeze(tmp_box,dim=0).detach().cpu()))
            final_scores=torch.cat((final_scores,scores[idx_box].detach().cpu().reshape(1)))
            final_labels=torch.cat((final_labels,class_labels[idx_box].detach().cpu().reshape(1)))
            
    return final_boxes,final_scores,final_labels

def correct_wrong_labels(id_box,boxes,scores,class_labels,booleans,page_id):
    box=boxes[id_box]

    cpt_overlaps=0
    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

        
        if ratio_a_in_b(tmp_box,box)>0.5 :
            
            cpt_overlaps+=1

            
    

    if cpt_overlaps==0 and class_labels[id_box]==6 and scores[id_box]<0.4:
        if torch.abs(box[3]-box[1])<250 and box[1]>250 :
            class_labels[id_box]=2

    if page_id==1 and class_labels[id_box]==6:
        if  max(box[3],box[1])<250 :
            class_labels[id_box]=5

    if cpt_overlaps==0 and (int(class_labels[id_box]) in [7,9]):
        if (min(box[2],box[0])>400 and max(box[1],box[3])<500) or min(box[1],box[3])<125  :
            class_labels[id_box]=5
            
    if class_labels[id_box]==3:
        class_labels[id_box]=9

    if class_labels[id_box]==2 and scores[id_box]<0.55 :
        class_labels[id_box]=6

    
        
            

    
        

    return class_labels
            
            
    

def supress_elements_inside_box(id_box,boxes,scores,class_labels,booleans):
    #this function supress text elements nested inside box 
    

    box=boxes[id_box]

    if booleans[id_box]==0:
        return booleans

    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

        
        if ratio_a_in_b(tmp_box,box)>0.80 and (scores[id_box]>scores[idx_box]  ) and class_labels[idx_box]!=0  :
            
            booleans[idx_box]=0.0

        #supress tables nestsed in tables
        if ratio_a_in_b(tmp_box,box)>0.80 and class_labels[id_box]==8 and class_labels[idx_box]==8 :
            booleans[idx_box]=0.0

        if ratio_a_in_b(tmp_box,box)>0.80 and class_labels[id_box]==8 and class_labels[idx_box]==9 :
            booleans[idx_box]=0.0

        if ratio_a_in_b(tmp_box,box)>0.99 and class_labels[id_box]==6  and class_labels[idx_box]!=0:
            booleans[idx_box]=0.0

        #supress captions inside captions:

        if ratio_a_in_b(tmp_box,box)>0.7 and class_labels[id_box]==0  and class_labels[idx_box]==0:
            booleans[idx_box]=0.0

        
        
    
            
            
    return booleans



def supress_dummies_boxes(id_box,boxes,scores,class_labels,booleans):
    #this function supress text elements nested inside box 
    

    box=boxes[id_box]

    if booleans[id_box]==0:
        return booleans

    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

        
        if ratio_a_in_b(tmp_box,box)>0.95 and scores[idx_box]>scores[id_box] and class_labels[id_box]!=8 and class_labels[id_box]!=6  :
            
            booleans[id_box]=0.0
            
    return booleans

def supress_dummies_tables(id_box,boxes,scores,class_labels,booleans):
    #this function supress text elements nested inside box 
    

    box=boxes[id_box]

    if booleans[id_box]==0:
        return booleans

    cpt=0

    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

        
        if ratio_a_in_b(tmp_box,box)>0.95 and scores[idx_box]>scores[id_box]:
            cpt+=1

    if cpt>=3:
        booleans[id_box]=0.0
            
    return booleans

def supress_pictures_with_nested_pictures(id_box,boxes,scores,class_labels,booleans):
    #this function supress text elements nested inside box 
    

    box=boxes[id_box]

    if booleans[id_box]==0:
        return booleans

    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

        
        if ratio_a_in_b(tmp_box,box)>0.80 and class_labels[id_box]==6 and class_labels[idx_box]==6 :
            
            booleans[id_box]=0.0
            
    return booleans

def recalage_pictures(id_box,boxes,scores,class_labels,booleans):
    if class_labels[id_box]!=6:
        return None

   

    b1_xa=min(boxes[id_box][0].item(),boxes[id_box][2].item())
    b1_ya=min(boxes[id_box][1].item(),boxes[id_box][3].item())
    b1_xb=max(boxes[id_box][0].item(),boxes[id_box][2].item())
    b1_yb=max(boxes[id_box][1].item(),boxes[id_box][3].item())

    b1_xc=(b1_xa+b1_xb)/2.0
    b1_yc=(b1_ya+b1_yb)/2.0
        
    #recherche d'une box de meme hateur
    for idx_box in range(boxes.shape[0]):
        if class_labels[idx_box]==6:
            b2_xa=min(boxes[idx_box][0].item(),boxes[idx_box][2].item())
            b2_ya=min(boxes[idx_box][1].item(),boxes[idx_box][3].item())
            b2_xb=max(boxes[idx_box][0].item(),boxes[idx_box][2].item())
            b2_yb=max(boxes[idx_box][1].item(),boxes[idx_box][3].item())
        
            b2_xc=(b2_xa+b2_xb)/2.0
            b2_yc=(b2_ya+b2_yb)/2.0

            if np.sqrt((b1_yc-b2_yc)**2)<100:
                if np.sqrt((b1_xa-b2_xb)**2)>1500 and np.sqrt((b1_xb-b2_xa)**2) < 250  :
                    min_x=min(b2_xa,b1_xa)
                    min_y=min(b2_ya,b1_ya)
                    
                    max_x=max(b2_xb,b1_xb)
                    max_y=max(b2_yb,b1_yb)

                    mid_x=(min_x+max_x)/2.0

                    idx_gauche=None
                    idx_droit=None

                    if b1_xa<b2_xa:
                        idx_gauche=id_box
                        idx_droit=idx_box
                    else:
                        idx_gauche=idx_box
                        idx_droit=id_box
                        
                    boxes[idx_gauche][0]=min_x
                    boxes[idx_gauche][1]=max_y
                    boxes[idx_gauche][2]=827.0
                    boxes[idx_gauche][3]=min_y

                    boxes[idx_gauche][0]=827.0
                    boxes[idx_gauche][1]=max_y
                    boxes[idx_gauche][2]=max_x
                    boxes[idx_gauche][3]=min_y
                    
    return boxes

def convert_to_captions(id_box,boxes,class_labels,img):
    test=extract_words_bb(boxes[id_box],img,show=False)

    
    
    if (("table " in test[:10].lower()) or ("fig" in test[:10].lower())) and len(test)<300:
        class_labels[id_box]=0

    
    if len(test)>200:
        class_labels[id_box]=9
        

    if len(test)>50 and class_labels[id_box]==5:
        class_labels[id_box]=9

    return class_labels
    

def resize_pictures_with_captions(id_box,boxes,scores,class_labels,booleans,img):
    if class_labels[id_box]!=6:
        return boxes

    

    box=boxes[id_box]
    w=np.sqrt((box[0]-box[2])**2)

    if w>1000:
        return boxes
        
    mid_y=(box[1]+box[3])/2.0

    for idx_box in range(boxes.shape[0]):
        if class_labels[idx_box]==0:
            tmp_box=boxes[idx_box]

            if ratio_a_in_b(tmp_box,box)>0.3:

                if tmp_box[3]>mid_y:
                    boxes[id_box][1]=tmp_box[3]-5
                else:
                    boxes[id_box][3]=tmp_box[1]+5
                
    return boxes

 

def find_hidden_captions(id_box,boxes,scores,class_labels,booleans,img,page_id,bottom_line=None,word="fig."):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y
    box=boxes[id_box]

    if bottom_line!=None:
        if box[1]>bottom_line:
            boxes[id_box][1]=bottom_line-10
        

    
    for idx_box in range(boxes.shape[0]):
        tmp_box=boxes[idx_box]

        if class_labels[idx_box]==0 and ratio_a_in_b(tmp_box,box)>0.70: 
            return boxes,scores,class_labels,booleans

    box=boxes[id_box]
    res_extraction_caption=extract_bb(box,img,show=False,erosion=False,word=word)
    
    if res_extraction_caption[0]!=None:
        
        y_min,y_max,x_min,x_max=res_extraction_caption
        tmp_newbox=torch.zeros((1,4))
        tmp_score=torch.empty(1)
        tmp_label=torch.empty(1)
        tmp_bool=torch.empty(1)
        tmp_newbox[0,0]=min(x_min,x_max)+int(box[0].item())
        tmp_newbox[0,1]=int(box[1].item())-max(y_max,y_min)
        tmp_newbox[0,2]=max(x_max,x_min)+int(box[0].item())
        tmp_newbox[0,3]=int(box[1].item())-min(y_min,y_max)

        boxes=torch.cat((boxes,tmp_newbox),dim=0)
        tmp_score=torch.empty(1)
        tmp_label=torch.empty(1)
        tmp_bool=torch.empty(1)
        
        tmp_score[0]=0.0
        tmp_label[0]=0
        tmp_bool[0]=1
        
        scores=torch.cat((scores,tmp_score),dim=0)
        class_labels=torch.cat((class_labels,tmp_label),dim=0)
        booleans=torch.cat((booleans,tmp_bool),dim=0)

        

        
        boxes=recalage_boxes_xgauche(boxes.shape[0]-1,boxes,img)
        boxes=recalage_boxes_yhaut(boxes.shape[0]-1,boxes,img)
        boxes=recalage_boxes_ybas(boxes.shape[0]-1,boxes,img)
        boxes=recalage_boxes_xdroit(boxes.shape[0]-1,boxes,img)

        #extract_words_bb(boxes[boxes.shape[0]-1],img,show=True,erosion=False)

    
    
        
        return boxes,scores,class_labels,booleans
    
    
    #if "fig." in tmp_text:
        
            
        

    return boxes,scores,class_labels,booleans




        


def correct_boxes_with_content(id_box,boxes,scores,class_labels,img,bottom_line,header_line):
    box=boxes[id_box]

    
    
    
    if class_labels[id_box] in [6,8]:

        if class_labels[id_box]==6:
            tmp_txt_eros=extract_words_bb(boxes[id_box],img,show=False,erosion=True)
    
            if "table" in tmp_txt_eros.lower():
                class_labels[id_box]=8
                return class_labels

        
        
        return class_labels
        

    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

        if ratio_a_in_b(box,tmp_box)>0.7 and class_labels[idx_box] in [6,8] :
            return class_labels
            

    

    tmp_txt=extract_words_bb(box,img,show=False,erosion=False)
    tmp_txt_erosion=extract_words_bb(box,img,show=False,erosion=True)

    
    

    if class_labels[id_box]==0:

        cond_a= (not "fig" in tmp_txt.lower()[:10] ) and (not "fig" in tmp_txt_erosion.lower()[:10] )
        cond_b= (not "table" in tmp_txt.lower()[:10] ) and (not "table" in tmp_txt_erosion.lower()[:10] )
        cond_c= ("shows" in tmp_txt.lower()) or ("illustrates" in tmp_txt.lower())

        width=np.sqrt((box[0]-box[2])**2)
        height=np.sqrt((box[1]-box[3])**2)

       
        if ((cond_a and cond_b ) or cond_c or height>200) and (scores[id_box]<0.95):
            if len(tmp_txt)<30 and (not "\n" in tmp_txt) and height<40:
                
                class_labels[id_box]=7
            else:
                class_labels[id_box]=9
            
    else:
        if "fig" in tmp_txt[:10].lower() or "table" in tmp_txt[:10].lower():
            class_labels[id_box]=0

        #wrong pade headers
        if class_labels[id_box]==5 and box[1]>=250:
            if not "vishay" in tmp_txt.lower() :
                class_labels[id_box]=7
            if  "esd level" in tmp_txt.lower():
                class_labels[id_box]=9

        #package marking wrongly assumed to be just text
        if "package" in tmp_txt[:7].lower() and len(tmp_txt)<150:
            class_labels[id_box]=7

        if class_labels[id_box]==7 and "vishay" in tmp_txt.lower() and box[1]<=500:
            class_labels[id_box]=5

        if class_labels[id_box]==7 and "logic" in tmp_txt.lower():
            class_labels[id_box]=9

        if class_labels[id_box]==7 and box[1]<=250:
            class_labels[id_box]=5


    

    
    if class_labels[id_box]==7 and boxes[id_box,0]>2100:
        class_labels[id_box]=6
        
    if (class_labels[id_box] in [9]) and boxes[id_box,1]<500:
        tmp_w=np.sqrt((boxes[id_box,0]-boxes[id_box,2])**2)
        if tmp_w>1500:
            class_labels[id_box]=5

    if bottom_line!=None and box[3]>=bottom_line and (not class_labels[id_box] in [6,8]):
        class_labels[id_box]=4

    if bottom_line!=None and box[1]<=header_line and (not class_labels[id_box] in [6,8]):
        class_labels[id_box]=5

    if class_labels[id_box]==7 and " it is " in tmp_txt:
        class_labels[id_box]=9

    extract_words_bb(boxes[id_box],img,show=False,erosion=False)
    if class_labels[id_box]==7 and ((" vs. " in tmp_txt) or (" vs. " in tmp_txt_erosion) or(" vs " in tmp_txt_erosion) or (" vs " in tmp_txt)  ):
        tmp_str="land pattern"
        if not ((tmp_str in tmp_txt.lower()) or (tmp_str in tmp_txt_erosion.lower())  ):
            
            class_labels[id_box]=0

    

    
    if bottom_line!=None and class_labels[id_box]==7 and boxes[id_box,3]>bottom_line:
        class_labels[id_box]=4

    box=boxes[id_box]
    h=np.sqrt((box[1].item()-box[3].item())**2)
    w=np.sqrt((box[1].item()-box[3].item())**2)

    if w<100 and h>500 and box[2]<500 and box[1]>2500:
        class_labels[id_box]=4
        
    

    

            
            

    
        
            

    
        

    return class_labels



def correct_sechead_height_end(head_coordinates_min,head_coordinates_max,id_box,boxes,scores,class_labels,img):
    box=boxes[id_box].clone()

    if not class_labels[id_box]==5:
        return boxes,scores,class_labels

    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y

    dist_haut=np.sqrt((head_coordinates_min-box[3])**2)>=20
    
    if box[1]>= head_coordinates_min-2 and dist_haut:
        if np.sqrt((head_coordinates_max-box[1])**2)<20:
            boxes[id_box,1]=head_coordinates_min-7
            scores[id_box]=2.5
        else:
            boxes[id_box,1]=head_coordinates_min-7
            scores[id_box]=2.5

            
            tmp_newbox=torch.zeros((1,4))
            tmp_score=torch.empty(1)
            tmp_label=torch.empty(1)
            
            tmp_newbox[0]=box.clone()
            tmp_newbox[0,3]=head_coordinates_max+7
            tmp_score[0]=1.5
            tmp_label[0]=5
            
            boxes=torch.cat((boxes,tmp_newbox),dim=0)
            scores=torch.cat((scores,tmp_score),dim=0)
            class_labels=torch.cat((class_labels,tmp_label),dim=0)
            
    return boxes,scores,class_labels

def correct_sechead_width_end(id_box,boxes,scores,class_labels,img):
    box=boxes[id_box].clone()

    if not class_labels[id_box]==5:
        return boxes,scores,class_labels

    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y

    width=np.sqrt((box[0]-box[2])**2)
    
    if width>=1500:
        boxes[id_box,2]=boxes[id_box,0]+width/2.0-1
        scores[id_box]=2.5
    
        
        tmp_newbox=torch.zeros((1,4))
        tmp_score=torch.empty(1)
        tmp_label=torch.empty(1)
        
        tmp_newbox[0]=box.clone()
        tmp_newbox[0,0]=boxes[id_box,0]+width/2.0+1
        tmp_score[0]=1.5
        tmp_label[0]=5
        
        boxes=torch.cat((boxes,tmp_newbox),dim=0)
        scores=torch.cat((scores,tmp_score),dim=0)
        class_labels=torch.cat((class_labels,tmp_label),dim=0)
                
    return boxes,scores,class_labels
                
                

def correct_sechead_bb(id_box,boxes,scores,class_labels,img):
    if class_labels[id_box]!=7:
        return boxes
    
    box=boxes[id_box]
   

    for tmp_idx_box in range(boxes.shape[0]):
        if id_box==tmp_idx_box:
            continue

        tmp_box=boxes[tmp_idx_box]

        if ratio_a_in_b(box,tmp_box)>0.80: 
            mid_y=(tmp_box[1]+tmp_box[3])/2.0

            if box[1]>mid_y:
                boxes[tmp_idx_box,1]=box[3]-2
            else:
                boxes[tmp_idx_box,3]=box[1]+2
        
                
    return boxes
                


    
def correct_caption_bb(id_box,boxes,scores,class_labels,img):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y
    tmp_box=boxes[id_box]

    
    if class_labels[id_box]==0:

        
        tmp_y=tmp_box[1]
        verif_blank=False
        for test_y in range(int(tmp_box[3].item()+2),int(tmp_box[1])):
            if 0 in img[test_y,int(tmp_box[0].item()+20):int(tmp_box[2].item()-20)]:
                verif_blank=True
            if verif_blank and( not 255 in img[test_y,int(tmp_box[0].item()+20):int(tmp_box[2].item()-20)]):
                boxes[id_box,1]=test_y-2
                break
                
        tmp_box=boxes[id_box]
        if not 0 in img[int(tmp_box[3].item()+3):int(tmp_box[1].item()-3),int(tmp_box[0].item()+7):int(tmp_box[0].item()+10)]:
            boxes[id_box,0]=int(tmp_box[0].item()+7)
            
        tmp_box=boxes[id_box]
        if not 0 in img[int(tmp_box[3].item()+3):int(tmp_box[1].item()-3),int(tmp_box[2].item()-20):int(tmp_box[2].item()-17)]:
            boxes[id_box,2]=int(tmp_box[2].item()-17)
    
        boxes=supress_blank_area(id_box,boxes,img)
  
    
            


    
    return boxes
        






def supress_boxes(id_box,boxes,scores,class_labels,booleans,page_id,img):

    if class_labels[id_box]==10:
        box=boxes[id_box]
        tmp_txt=extract_words_bb(box,img,show=False,erosion=False)

        if " unless " in tmp_txt:
            booleans[id_box]=0
            return booleans
            
        
    
    if class_labels[id_box]==6:
        w= np.sqrt((boxes[id_box,0].item()-boxes[id_box,2].item())**2)
        h= np.sqrt((boxes[id_box,1].item()-boxes[id_box,3].item())**2)

        
        if w<=75 or h<=75:
            booleans[id_box]=0

        box=boxes[id_box]
        if page_id==1 and w<200 and box[2]>2250 and h>200:
            booleans[id_box]=0

        
            

        

        for tmp_idx_box in range(boxes.shape[0]):
            if tmp_idx_box==id_box:
                continue
            
            if (not class_labels[tmp_idx_box] in [0,6]) and ratio_a_in_b(boxes[tmp_idx_box],boxes[id_box])>0.90 and scores[tmp_idx_box]>0.6:
                booleans[id_box]=0
                return booleans
                
        return booleans

    if class_labels[id_box]==8:
        if scores[id_box]<0.4:
            booleans[id_box]=0
        return booleans

    
            
    verif=False
    cpt_in_tables=0
    if class_labels[id_box]==0:
        
        box=boxes[id_box]
        h= np.sqrt((box[1].item()-box[3].item())**2)
        w= np.sqrt((box[0].item()-box[2].item())**2)
        
        tmp_txt=extract_words_bb(box,img,show=False,erosion=False)
        tmp_txt_eros=extract_words_bb(box,img,show=False,erosion=True)
        
        if w<50 or (len(tmp_txt)<1 and len(tmp_txt_eros)<1):
            booleans[id_box]=0
            return booleans
            
        for tmp_idx in range(boxes.shape[0]):
            if tmp_idx==id_box:
                continue

            if (not class_labels[tmp_idx] in [6,8]) and ratio_a_in_b(boxes[id_box],boxes[tmp_idx])>0.90 :
                verif=True

               
            if class_labels[tmp_idx]==8 and ratio_a_in_b(boxes[id_box],boxes[tmp_idx])>0.80 :
                cpt_in_tables+=1
                
                if np.sqrt((boxes[tmp_idx,0]-boxes[id_box,0])**2)>200:
                    booleans[id_box]=0
                    return booleans
        
        if verif and cpt_in_tables==0:
            booleans[id_box]=0
            return booleans
            

    if class_labels[id_box] == 5 :
        box=boxes[id_box]
        h= np.sqrt((box[1].item()-box[3].item())**2)
        
        if h>120 :
            
            cond_blank= not ( 0 in img[int(box[3].item()+25):int(box[1].item()-25),int(box[0].item()):int(box[2].item())])
            
            if cond_blank:
                booleans[id_box]=0
                return booleans

        
        for tmp_idx in range(boxes.shape[0]):
            if tmp_idx==id_box:
                continue

            #check if its blank
            #res[idx_box][0]=min_x
            #res[idx_box][1]=max_y
            #res[idx_box][2]=max_x
            #res[idx_box][3]=min_y
            tmp_box=boxes[tmp_idx]

            
            if ( class_labels[tmp_idx] in [8]) and ratio_a_in_b(boxes[id_box],boxes[tmp_idx])>0  :
                booleans[id_box]=0
                return booleans
                
                
    if class_labels[id_box]==7:
        box=boxes[id_box]
        tmp_txt=extract_words_bb(box,img,show=False,erosion=False)
        tmp_txt_eros=extract_words_bb(box,img,show=False,erosion=True)

        cond_txt_a="bottom view" in tmp_txt.lower()
        cond_txt_b="top view" in tmp_txt.lower()
        cond_txt_c="side view" in tmp_txt.lower()
                    
        if ( cond_txt_a or cond_txt_b or cond_txt_c  ) or (len(tmp_txt.replace(" ",""))<=1 and len(tmp_txt_eros.replace(" ",""))<=1   ):
            booleans[id_box]=0
            return booleans

        
      
    
    if class_labels[id_box] != 9 :
        return booleans

    box=boxes[id_box]
    tmp_txt=extract_words_bb(box,img,show=False,erosion=False)

    if page_id==1:
        if scores[id_box]<0.6:
            booleans[id_box]=0

    if len(tmp_txt)<=2:
        tmp_txt_erosion=extract_words_bb(box,img,show=False,erosion=True)

        if len(tmp_txt_erosion)>len(tmp_txt):
            tmp_txt=tmp_txt_erosion
   
    if (("view" in tmp_txt.lower()  and len(tmp_txt)<60)) or len(tmp_txt.replace(" ",""))<=1:
        booleans[id_box]=0

    
                    

    return booleans




def remove_unlabelled_text(boxes,img):
    model=torch.zeros(img.shape)

    for id_box in range(boxes.shape[0]):
        box=boxes[id_box]
        model[int(box[3].item()):int(box[1].item()),int(box[0].item()):int(box[2].item())]=1.0

    img[model==0]=0
    return model,img
    

            


def check_bugs(boxes,scores,class_labels,img):

    cpt_images=0
    cpt_captions=0
    cpt_section_headers=0
    L_section_headers=[]
    
    for id in range(boxes.shape[0]):
        box=boxes[id]

        if class_labels[id]==6:
            cpt_images+=1

        if class_labels[id]==7:
            cpt_section_headers+=1

            L_section_headers.append(extract_words_bb(box,img,show=False,erosion=True))
            L_section_headers.append(extract_words_bb(box,img,show=False,erosion=False))
            

        if class_labels[id]==0:
            cpt_captions+=1
            
        for tmp_idx in range(boxes.shape[0]):
            if id==tmp_idx:
                continue
            tmp_box=boxes[tmp_idx]

            tmp_r=ratio_a_in_b(box,tmp_box)

            if tmp_r>0:
                if class_labels[id]==6 and class_labels[tmp_idx]==7:
                    return False

                if class_labels[id]==4 and class_labels[tmp_idx]==7:
                    return False

                if class_labels[id]==6 and class_labels[tmp_idx]==6:
                    return False

    if cpt_images>=5 and cpt_section_headers==1:
        for tmp_txt in L_section_headers:
            if "characteristics" in tmp_txt.lower() and cpt_captions!=cpt_images:
                return False

            if "waveforms" in tmp_txt.lower() and cpt_captions!=cpt_images:
                return False

    return True

    
                
            
        

def box_fusion(f_b,f_s,f_l,page_id,img,fun):
    while(True):
        cpt_verifs=0

        for idx_box in range(f_b.shape[0]):

            if idx_box>=f_b.shape[0]:
                break

            tmp_bools=torch.ones(f_b.shape[0])
        
            verif,f_b,tmp_bools,f_l=fun(idx_box,f_b,f_s,f_l,tmp_bools,page_id,img)

            if verif:

                cpt_verifs+=1

                f_b=f_b[tmp_bools.to(torch.bool)]
                f_s=f_s[tmp_bools.to(torch.bool)]
                f_l=f_l[tmp_bools.to(torch.bool)]
                
        if cpt_verifs==0:
            break
    return f_b,f_s,f_l



            
  
def traitement_boxes(boxes,scores,class_labels,page_id,img):
    
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=np.array(img.copy())
    img[img<200]=0
    img[img>=200]=255
    
    
    f_b,f_s,f_l=supress_low_confidence_boxes(boxes,scores,class_labels)

    tmp_bools=torch.ones(f_b.shape[0])

    L_header=get_line_header_VISHAY(img,500)
   
    
    L_footer=get_line_footer_VISHAY(img,3000)

    bottom_line=None
    header_line=None
    if len(L_footer)!=0:
        bottom_line=max(L_footer)

    if len(L_header)!=0:
        header_line=max(L_header)
    
    for idx_box in range(f_b.shape[0]):
        tmp_bools=supress_pictures_with_nested_pictures(idx_box,f_b,f_s,f_l,tmp_bools)
        
    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    tmp_bools=torch.ones(f_b.shape[0])
    
    for idx_box in range(f_b.shape[0]):
        tmp_bools=supress_elements_inside_box(idx_box,f_b,f_s,f_l,tmp_bools)




    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    tmp_bools=torch.ones(f_b.shape[0])
    for idx_box in range(f_b.shape[0]):
        f_l=correct_wrong_labels(idx_box,f_b,f_s,f_l,tmp_bools,page_id)
        
        tmp_bools=supress_dummies_boxes(idx_box,f_b,f_s,f_l,tmp_bools)



    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    

    

    for idx_box in range(f_b.shape[0]):
        if not (f_l[idx_box].to(int).item()) in [6,8]:
            f_b=recalage_boxes_xgauche(idx_box,f_b,img,step=5)
            f_b=recalage_boxes_xdroit(idx_box,f_b,img,step=5)
            f_b=recalage_boxes_yhaut(idx_box,f_b,img,step=5)
            f_b=recalage_boxes_ybas(idx_box,f_b,img,step=5)
           

            f_l=convert_to_captions(idx_box,f_b,f_l,img)
        else:
            
            if f_b[idx_box,1]>=750 and page_id>1:
                f_b=recalage_boxes_xgauche(idx_box,f_b,img,step=10,attempt=True)
                f_b=recalage_boxes_ybas(idx_box,f_b,img,step=3)
            else:
                f_b=recalage_boxes_xgauche(idx_box,f_b,img,step=1)
                f_b=recalage_boxes_ybas(idx_box,f_b,img,step=1)
                
            f_b=recalage_boxes_xdroit(idx_box,f_b,img,step=1)
            f_b=recalage_boxes_yhaut(idx_box,f_b,img,step=1)
            


    
    
    for idx_box in range(f_b.shape[0]):
        if  (f_l[idx_box].to(int).item())==6:
            f_b=resize_pictures_with_captions(idx_box,f_b,f_s,f_l,tmp_bools,img)


    tmp_bools=torch.ones(f_b.shape[0])
    
    for idx_box in range(f_b.shape[0]):
        f_l,tmp_bools=correct_wrong_labels_STMICRO(idx_box,f_b,f_s,f_l,tmp_bools,img,page_id,bottom_line,header_line)

    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    
    

    for idx_box in range(f_b.shape[0]):
        if  (f_l[idx_box].to(int).item())==8:
           
            f_b,f_s,f_l,tmp_bools=find_captions_table_STMICRO(idx_box,f_b,f_s,f_l,tmp_bools,img)

    for idx_box in range(f_b.shape[0]):
        if  (f_l[idx_box].to(int).item())==6:
            f_b,f_s,f_l,tmp_bools=find_captions_figures_STMICRO(idx_box,f_b,f_s,f_l,tmp_bools,img)

    
    
            
            


    
    


    for idx_box in range(f_b.shape[0]):
        if  (f_l[idx_box].to(int).item())==6:
            f_b,f_s,f_l,tmp_bools=find_hidden_captions(idx_box,f_b,f_s,f_l,tmp_bools,img,page_id,bottom_line,word="fig.")

            
    for idx_box in range(f_b.shape[0]):
        if  (f_l[idx_box].to(int).item())==6:
            f_b=resize_pictures_with_captions(idx_box,f_b,f_s,f_l,tmp_bools,img)

    
    

    tmp_bools=torch.ones(f_b.shape[0])
    
    for idx_box in range(f_b.shape[0]):
        if  (f_l[idx_box].to(int).item())==8:
            tmp_bools=supress_dummies_tables(idx_box,f_b,f_s,f_l,tmp_bools)

    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]


    


    

    
     
        
    for idx_box in range(f_b.shape[0]):
        f_l=correct_boxes_with_content(idx_box,f_b,f_s,f_l,img,bottom_line,header_line)


 

    f_b,f_s,f_l=box_fusion(f_b,f_s,f_l,page_id,img,supress_overlapping_boxes_STMICRO)

    


    
    
    
    

    if len(L_header)!=0:
        for idx_box in range(f_b.shape[0]):
            f_b,f_s,f_l=correct_sechead_height_end(min(L_header),max(L_header),idx_box,f_b,f_s,f_l,img)


    coord_line_deb=None
    
    

    
    
    

    for idx_box in range(f_b.shape[0]):
        f_b=supress_blank_area(idx_box,f_b,img)

    for idx_box in range(f_b.shape[0]):
        f_b,f_s,f_l=correct_sechead_width_end(idx_box,f_b,f_s,f_l,img)

    for idx_box in range(f_b.shape[0]):
        f_b=supress_blank_area(idx_box,f_b,img)
    
    for idx_box in range(f_b.shape[0]):
        correct_sechead_bb(idx_box,f_b,f_s,f_l,img)

    for idx_box in range(f_b.shape[0]):
        f_b=supress_blank_area(idx_box,f_b,img)

    


    for idx_box in range(f_b.shape[0]):
        f_b=correct_caption_bb(idx_box,f_b,f_s,f_l,img)

    

    for idx_box in range(f_b.shape[0]):
        if f_l[idx_box] in [6,8]:
            f_b=recalage_boxes_yhaut(idx_box,f_b,img,step=1)
            f_b=recalage_boxes_ybas(idx_box,f_b,img,step=1)

    
            

    


    

    
    while True:
        verif=False
        for idx_box in range(f_b.shape[0]):

            box=f_b[idx_box]
            w=np.sqrt((box[0]-box[2])**2)
            h=np.sqrt((box[1]-box[3])**2)

            if f_l[idx_box]==6:

                if h<150 :
                    tmp_verif,f_b,f_s,f_l=divide_blank_space_vertical(idx_box,f_b,f_s,f_l,img,min_width=30,mw_nb=75)
                    verif=verif or tmp_verif
    
                    if tmp_verif:
                        f_b=supress_blank_area(idx_box,f_b,img)
                        f_b=supress_blank_area(f_b.shape[0]-1,f_b,img)
    
                if w < 150 and h>90 :
                    tmp_verif,f_b,f_s,f_l=divide_blank_space_horizontal(idx_box,f_b,f_s,f_l,img,min_width=30,mw_nb=30)
                    verif=verif or tmp_verif
    
                    if tmp_verif:
                        f_b=supress_blank_area(idx_box,f_b,img)
                        f_b=supress_blank_area(f_b.shape[0]-1,f_b,img)
                continue

            
            
                
                
        if not verif:
            break

    


    


    
    
    for idx_box in range(f_b.shape[0]):
        f_b=supress_blank_area(idx_box,f_b,img)





    
    tmp_bools=torch.ones(f_b.shape[0])

 
    
    for idx_box in range(f_b.shape[0]):
        tmp_bools=supress_boxes_STMICRO(idx_box,f_b,f_s,f_l,tmp_bools,page_id,img,bottom_line,header_line)


    


  
    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    

    f_b,f_s,f_l=box_fusion(f_b,f_s,f_l,page_id,img,supress_overlapping_boxes_STMICRO)

    for idx_box in range(f_b.shape[0]):
        if (f_l[idx_box].to(int).item()) in [6,8]:
            f_b=recalage_boxes_xgauche(idx_box,f_b,img,step=1,attempt=True)
            f_b=recalage_boxes_xdroit(idx_box,f_b,img,step=1,attempt=True)
            f_b=recalage_boxes_yhaut(idx_box,f_b,img,step=1,attempt=True)
            f_b=recalage_boxes_ybas(idx_box,f_b,img,step=1,attempt=True)

            f_b=supress_blank_area(idx_box,f_b,img)




    

   

    for idx_box in range(f_b.shape[0]):
        f_b,f_s,f_l=correction_hidden_images_STMICRO(idx_box,f_b,f_s,f_l,img)


   
    

    for idx_box in range(f_b.shape[0]):
        if not (f_l[idx_box].to(int).item()) in [6,8]:
            f_b=recalage_boxes_xgauche(idx_box,f_b,img,step=15,attempt=True)
            f_b=recalage_boxes_xdroit(idx_box,f_b,img,step=15,attempt=True)
            f_b=recalage_boxes_yhaut(idx_box,f_b,img,step=2,attempt=True)
            f_b=recalage_boxes_ybas(idx_box,f_b,img,step=2,attempt=True)

            f_b=supress_blank_area(idx_box,f_b,img)

    tmp_bools=torch.ones(f_b.shape[0])

    for idx_box in range(f_b.shape[0]):

        f_b,f_s,f_l,tmp_bools=suppress_blank_boxes(idx_box,f_b,f_s,f_l,img,tmp_bools)

    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    
    tmp_bools=torch.ones(f_b.shape[0])

    for idx_box in range(f_b.shape[0]):
        f_b,f_s,f_l,tmp_bools=last_correction_overlapping_captions(idx_box,f_b,f_s,f_l,img,tmp_bools)

    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    tmp_bools=torch.ones(f_b.shape[0])

    for idx_box in range(f_b.shape[0]):
        f_b,f_s,f_l,tmp_bools=last_correction_text_captions(idx_box,f_b,f_s,f_l,img,tmp_bools)

    f_b=f_b[tmp_bools.to(torch.bool)]
    f_s=f_s[tmp_bools.to(torch.bool)]
    f_l=f_l[tmp_bools.to(torch.bool)]

    


    for idx_box in range(f_b.shape[0]):
        f_b=last_correction_sec_head_left_side(idx_box,f_b,f_s,f_l,img)

    f_b,f_s,f_l=box_fusion(f_b,f_s,f_l,page_id,img,supress_overlapping_boxes_STMICRO)

    for idx_box in range(f_b.shape[0]):
        if (f_l[idx_box].to(int).item()) in [6,8]:
            f_b=recalage_boxes_yhaut(idx_box,f_b,img,step=1,attempt=True)
            f_b=recalage_boxes_ybas(idx_box,f_b,img,step=1,attempt=True)

            f_b=supress_blank_area(idx_box,f_b,img)

    

    for idx_box in range(f_b.shape[0]):
        if not (f_l[idx_box].to(int).item()) in [6,8]:
            f_b=recalage_boxes_xgauche(idx_box,f_b,img,step=15,attempt=True)
            f_b=recalage_boxes_xdroit(idx_box,f_b,img,step=15,attempt=True)
            f_b=recalage_boxes_yhaut(idx_box,f_b,img,step=2,attempt=True)
            f_b=recalage_boxes_ybas(idx_box,f_b,img,step=2,attempt=True)

            f_b=supress_blank_area(idx_box,f_b,img)

    
                    




    

    


    while True:
        verif=False
        for idx_box in range(f_b.shape[0]):

            box=f_b[idx_box]
            w=np.sqrt((box[0]-box[2])**2)
            h=np.sqrt((box[1]-box[3])**2)



            if f_l[idx_box] in  [4,5]:

                
                
                
                tmp_verif,f_b,f_s,f_l=divide_blank_space_horizontal(idx_box,f_b,f_s,f_l,img,min_width=5,mw_nb=5)
                verif=verif or tmp_verif
    
                if tmp_verif:
                    f_b=supress_blank_area(idx_box,f_b,img)
                    f_b=supress_blank_area(f_b.shape[0]-1,f_b,img)
                continue
                        
            
                
                
        if not verif:
            break

    for idx_box in range(f_b.shape[0]):
        tmp_bools=supress_boxes_STMICRO(idx_box,f_b,f_s,f_l,tmp_bools,page_id,img,bottom_line,header_line)

    
    
    
    for idx_box in range(f_b.shape[0]):
        box=f_b[idx_box]
        if f_l[idx_box]==6 and box[3]<=250:
            if box[2]<1000:
                coord_line_deb=box[2]+50
            else:
                coord_line_deb=box[0]-50


    



    model,img=remove_unlabelled_text(f_b,img)
    drawing_utils=(model,L_header,coord_line_deb,L_footer)

    #return check_bugs(f_b,f_s,f_l,img),f_b,f_s,f_l,drawing_utils
    return True,f_b,f_s,f_l,drawing_utils
