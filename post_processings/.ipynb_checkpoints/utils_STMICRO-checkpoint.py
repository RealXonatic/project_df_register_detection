import torch
import matplotlib.pyplot as plt
import pytesseract
import cv2 as cv
import numpy as np

from .OCR_extraction import *
from .boxes import * 
from .box_utils import *




def find_captions_table_STMICRO(id_box,boxes,scores,class_labels,booleans,img):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y
    
   
        
    box=boxes[id_box]
    caption_in=False


    for idx_box in range(boxes.shape[0]):
        tmp_box=boxes[idx_box]
        if class_labels[idx_box]==0 and ratio_a_in_b(tmp_box,box)>0.98:
            caption_in=True

    #find y level box
    tmp_y_haut=int(box[3].item())
    tmp_y_bas=int(box[3].item())
    tmp_x_droit=int(box[2].item()-50)
    tmp_x_gauche=int(box[0].item()+50)

    if caption_in==False:
        tmp_y_bas=tmp_y_bas-75
        tmp_y_haut=tmp_y_haut-75

    cpt_line=0
    cpt_interline=0
    while cpt_line<=21 and tmp_y_bas<img.shape[0]:
        if not (255 in img[tmp_y_bas,tmp_x_gauche:tmp_x_droit]):
            break
            
        tmp_y_bas+=1

    
    
    tmp_newbox=torch.zeros((1,4))
    tmp_newbox[0,0]=int(box[0].item())
    tmp_newbox[0,1]=tmp_y_bas
    tmp_newbox[0,2]=int(box[2].item())
    tmp_newbox[0,3]=tmp_y_haut


    
    y_min,y_max,x_min,x_max=extract_bb_reforged(tmp_newbox[0],img,show=False,erosion=True,word="Table",tol=40)

    
    if y_min==None:
        y_min,y_max,x_min,x_max=extract_bb_reforged(tmp_newbox[0],img,show=False,erosion=False,word="Table",tol=40)
        
    else:
        y_min_eros,y_max_eros,x_min_eros,x_max_eros=extract_bb_reforged(tmp_newbox[0],img,show=False,erosion=False,word="Table")
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
    tmp_newbox[0,1]=y_min+width*2+int(box[3].item())-75
    tmp_newbox[0,2]=mid_x+width_x+int(box[0].item())
    tmp_newbox[0,3]=y_min+width+int(box[3].item())-75

    
            



    
        

    

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
    


    
def find_captions_figures_STMICRO(id_box,boxes,scores,class_labels,booleans,img,tol=75,only_eros=False):
    box=boxes[id_box]
    
    tmp_newbox=torch.zeros((1,4))
    tmp_newbox[0,0]=int(box[0].item())
    tmp_newbox[0,1]=int(box[1].item())+75
    tmp_newbox[0,2]=int(box[2].item())
    tmp_newbox[0,3]=int(box[3].item())-75


    y_min=None
    y_max=None
    x_min=None
    x_max=None

    if not only_eros:
        y_min,y_max,x_min,x_max=extract_bb_reforged(tmp_newbox[0],img,show=False,erosion=False,word="figure",tol=tol)
    
    
    if y_min==None:
        y_min,y_max,x_min,x_max=extract_bb_reforged(tmp_newbox[0],img,show=False,erosion=True,word="figure",tol=tol)
    else:
        y_min_eros,y_max_eros,x_min_eros,x_max_eros=extract_bb_reforged(tmp_newbox[0],img,show=False,erosion=True,tol=tol)
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
    tmp_newbox[0,1]=y_min+width*2+int(box[3].item())-75
    tmp_newbox[0,2]=mid_x+width_x+int(box[0].item())
    tmp_newbox[0,3]=y_min+width+int(box[3].item())-75

    boxes=torch.cat((boxes,tmp_newbox),dim=0)

    
    
    boxes=recalage_boxes_yhaut(boxes.shape[0]-1,boxes,img,step=1)
    boxes=recalage_boxes_ybas(boxes.shape[0]-1,boxes,img,step=1)
    boxes=recalage_boxes_xgauche(boxes.shape[0]-1,boxes,img)
    boxes=recalage_boxes_xdroit(boxes.shape[0]-1,boxes,img)
    boxes=supress_blank_area(boxes.shape[0]-1,boxes,img)

    
        
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
    
    


    

def correct_wrong_labels_STMICRO(id_box,boxes,scores,class_labels,booleans,img,page_id,bottom_line,header_line):
    box=boxes[id_box]

    cpt_overlaps=0
    in_table=False
    in_picture=False

    
    for idx_box in range(boxes.shape[0]):

        if id_box==idx_box:
            continue
            
        tmp_box=boxes[idx_box]

        
        if ratio_a_in_b(tmp_box,box)>0.90 and class_labels[idx_box]==7 and class_labels[id_box]==8:
            class_labels[idx_box]=0
            

        if ratio_a_in_b(tmp_box,box)>0.99 and class_labels[idx_box]==9 and class_labels[id_box]==8:
            class_labels[idx_box]=0

        if ratio_a_in_b(box,tmp_box)>0.99 and class_labels[idx_box]==8:
            in_table=True

        if ratio_a_in_b(box,tmp_box)>0.99 and class_labels[idx_box]==6:
            in_picture=True

    tmp_txt=extract_words_bb(box,img).lower()

    
    
    if  "note " in tmp_txt.lower()[:10] or "note:" in tmp_txt.lower()[:10] or "notes " in tmp_txt.lower()[:10] or "notes:" in tmp_txt.lower()[:10] :
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
            
        

    tmp_txt=extract_words_bb(box,img,show=False).lower()
    tmp_txt_eros=extract_words_bb(box,img,erosion=True,show=False).lower()
    
    if class_labels[id_box]==7 and (("table" in tmp_txt) or ("fig" in tmp_txt)):
        
        class_labels[id_box]=0

        

    if class_labels[id_box]==7 and (("table" in tmp_txt_eros) or ("fig" in tmp_txt_eros)):
        
        class_labels[id_box]=0
        
    
    if bottom_line!=None and box[3]>=bottom_line and (not class_labels[id_box] in [6,8]):
        class_labels[id_box]=4

    if class_labels[id_box]==7 and extract_words_bb(box,img).lower()[:4]=="www." and box[1].item()<=500:
        class_labels[id_box]=5

    if class_labels[id_box]==6:
        tmp_txt=extract_words_bb(box,img,show=False,erosion=True)

        if "information" in tmp_txt.lower():
            class_labels[id_box]=8

        if "dimensions" in tmp_txt.lower():
            class_labels[id_box]=8

    if class_labels[id_box]==0 and (in_table or in_picture):
        tmp_txt=extract_words_bb(box,img,show=False,erosion=True)

        if (not "table" in tmp_txt.lower()) and (not "fig" in tmp_txt[:10].lower()):
            booleans[id_box]=0

    if class_labels[id_box]==0:

        

        if ("shows" in tmp_txt.lower()) or ("illustrates" in tmp_txt.lower()):
            class_labels[id_box]=9

    
            
            


    
        
            

    
        

    return class_labels,booleans
            


def supress_overlapping_boxes_STMICRO(id_box,boxes,scores,class_labels,booleans,page_id,img):
    #this function supress text elements nested inside box 
    

    box=boxes[id_box]

    if booleans[id_box]==0:
        return False,boxes,booleans,class_labels

    if class_labels[id_box].item() in [6,8]:

        
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
                

            
    
            if class_labels[idx_box]==0 and ratio_a_in_b(tmp_box,box)>0.90:
                mid_y_table=(box[1]+box[3])/2.0

                if tmp_box[3]<mid_y_table and tmp_box[1]>mid_y_table:
                    continue
    
                if tmp_box[3]<mid_y_table:
                    boxes[id_box,3]=tmp_box[1]+3
                else:
                    boxes[id_box,1]=tmp_box[3]-3

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
        
        

        #box or picture nested inside text
        cond_e= ratio_a_in_b(tmp_box,box)>0.30 and class_labels[id_box]==9 and (class_labels[idx_box]==6 or class_labels[idx_box]==8 )

   
        
        if cond_e:
            booleans[id_box]=False
            verif=True
            return verif,boxes,booleans,class_labels

        #caption nested with sec_head
        if ratio_a_in_b(tmp_box,box)>0.30 and class_labels[id_box]==7 and class_labels[idx_box]==0 and scores[id_box]>0.95:
            booleans[idx_box]=False
            verif=True
            return verif,boxes,booleans,class_labels
            

        

        
        
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





def supress_boxes_STMICRO(id_box,boxes,scores,class_labels,booleans,page_id,img,bottom_line,header_line):

    w= np.sqrt((boxes[id_box,0].item()-boxes[id_box,2].item())**2)
    h= np.sqrt((boxes[id_box,1].item()-boxes[id_box,3].item())**2)

    if h>50 and w<10 and class_labels[id_box]!=6:
        booleans[id_box]=0
        return booleans

    if header_line!=None and class_labels[id_box]==5 and max(boxes[id_box,3].item(),boxes[id_box,1].item())>header_line:
        booleans[id_box]=0
        return booleans

    if class_labels[id_box]==4 and w>1250:
        booleans[id_box]=0
        return booleans

    if class_labels[id_box]==2:
        box=boxes[id_box]

        for tmp_idx in range(boxes.shape[0]):
            if tmp_idx==id_box:
                continue

            if ratio_a_in_b(boxes[tmp_idx],boxes[id_box])>0.90 or ratio_a_in_b(boxes[id_box],boxes[tmp_idx])>0.90 :
                booleans[tmp_idx]=0
                booleans[id_box]=0
                return booleans


        if scores[id_box]<0.4:
            booleans[id_box]=0
        return booleans
        
    

    

    if class_labels[id_box]==10:
        box=boxes[id_box]
        tmp_txt=extract_words_bb(box,img,show=False,erosion=False)

        if " unless " in tmp_txt:
            booleans[id_box]=0
            return booleans


    
    
    if class_labels[id_box]==0:
        tmp_txt=extract_words_bb(boxes[id_box],img,show=False,erosion=False)
        tmp_txt_eros=extract_words_bb(boxes[id_box],img,show=False,erosion=True)
        
        

        tmp_txt=extract_words_bb(boxes[id_box],img,show=False,erosion=False)

       
        if (h<5 or w<30 ) or  (len(tmp_txt)<2 and len(tmp_txt_eros)<2):
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

        if  box[3]>3000 and w>500:
            booleans[id_box]=0

        
            

        

        for tmp_idx_box in range(boxes.shape[0]):
            if tmp_idx_box==id_box:
                continue

            
            
            if (not class_labels[tmp_idx_box] in [0,6]) and ratio_a_in_b(boxes[tmp_idx_box],boxes[id_box])>0.90 and scores[tmp_idx_box]>0.6:
                booleans[id_box]=0
                return booleans
                
        return booleans

    if class_labels[id_box]==8:
        box=boxes[id_box]

        if  box[3]>3000 and w>500:
            booleans[id_box]=0
            return booleans
            
        for tmp_idx in range(boxes.shape[0]):
            if tmp_idx==id_box:
                continue

            if (not class_labels[tmp_idx] in [6,8,0]) and ratio_a_in_b(boxes[tmp_idx],boxes[id_box])>0.90 :
                booleans[tmp_idx]=0
                return booleans


        if scores[id_box]<0.4:
            booleans[id_box]=0
        return booleans

    
            
    verif=False
    

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




def correction_hidden_images_STMICRO(id_box,boxes,scores,class_labels,img):
    if class_labels[id_box]!=6:
        return boxes,scores,class_labels

    box=boxes[id_box]
    
    tmp_txt=extract_words_bb(box,img,show=False,erosion=False)
    
    if tmp_txt.lower().count("figure")>=2:
        
        mid_x=(box[0]+box[2])/2.0

        tmp_box=box.clone()
        box[2]=mid_x+1
        tmp_box[0]=mid_x-1

        boxes,scores,class_labels=add_box(tmp_box,6,1.50,boxes,scores,class_labels)

        booleans=torch.ones(boxes.shape[0])

        
        
        boxes,scores,class_labels,booleans=find_captions_figures_STMICRO(id_box,boxes,scores,class_labels,booleans,img,tol=50,only_eros=True)
        h_new_cap=np.sqrt((boxes[boxes.shape[0]-1,1].item()-boxes[boxes.shape[0]-1,3].item())**2)
        h_box=np.sqrt((boxes[id_box,1].item()-boxes[id_box,3].item())**2)
        
        boxes[boxes.shape[0]-1,0]=max(boxes[boxes.shape[0]-1,0],box[0]+20)
        boxes[boxes.shape[0]-1,1]=boxes[boxes.shape[0]-1,1]+75
        boxes[boxes.shape[0]-1,2]=min(boxes[boxes.shape[0]-1,2],box[2]-30)
        boxes[boxes.shape[0]-1,3]=max(boxes[boxes.shape[0]-1,3],box[3]+20)

                
        
        
        if np.sqrt((h_new_cap-h_box)**2)<40:
            booleans[id_box]=0
            booleans[boxes.shape[0]-1]=0

            boxes=boxes[booleans.to(torch.bool)]
            scores=scores[booleans.to(torch.bool)]
            class_labels=class_labels[booleans.to(torch.bool)]
        else:
            
            verif,boxes,scores,class_labels=divide_blank_space_horizontal(boxes.shape[0]-1,boxes,scores,class_labels,img,min_width=10,mw_nb=10)
            
            if verif:
                

          

                box_caption_haut=None
                box_caption_bas=None
    
                if boxes[boxes.shape[0]-2,3]<boxes[boxes.shape[0]-1,3]:
                    box_caption_haut=boxes.shape[0]-2
                    box_caption_bas=boxes.shape[0]-1
                else:
                    box_caption_haut=boxes.shape[0]-1
                    box_caption_bas=boxes.shape[0]-2

                



        booleans=torch.ones(boxes.shape[0])
        
        boxes,scores,class_labels,booleans=find_captions_figures_STMICRO(boxes.shape[0]-2,boxes,scores,class_labels,booleans,img,tol=50,only_eros=True)
        h_new_cap=np.sqrt((boxes[boxes.shape[0]-1,1].item()-boxes[boxes.shape[0]-1,3].item())**2)

        boxes[boxes.shape[0]-1,0]=max(boxes[boxes.shape[0]-1,0],boxes[boxes.shape[0]-3,0]+20)
        boxes[boxes.shape[0]-1,1]=boxes[boxes.shape[0]-1,1]+75
        boxes[boxes.shape[0]-1,2]=min(boxes[boxes.shape[0]-1,2],boxes[boxes.shape[0]-3,2]-30)
        boxes[boxes.shape[0]-1,3]=max(boxes[boxes.shape[0]-1,3],boxes[boxes.shape[0]-3,3]+20)
    
        verif=False
        if np.sqrt((h_new_cap-h_box)**2)<10:
            booleans[id_box]=0
            booleans[boxes.shape[0]-1]=0

            boxes=boxes[booleans.to(torch.bool)]
            scores=scores[booleans.to(torch.bool)]
            class_labels=class_labels[booleans.to(torch.bool)]
            
        else:
            
            if verif:
                

          

                box_caption_haut=None
                box_caption_bas=None
    
                if boxes[boxes.shape[0]-2,3]<boxes[boxes.shape[0]-1,3]:
                    box_caption_haut=boxes.shape[0]-2
                    box_caption_bas=boxes.shape[0]-1
                else:
                    box_caption_haut=boxes.shape[0]-1
                    box_caption_bas=boxes.shape[0]-2

 

        
        
        
        

    return boxes,scores,class_labels

def last_correction_overlapping_captions(id_box,boxes,scores,class_labels,img,tmp_bools):
    box=boxes[id_box]
   
    if class_labels[id_box]!=0 or tmp_bools[id_box]==0:
        return boxes,scores,class_labels,tmp_bools
        
    
    h_box=np.sqrt((box[1].item()-box[3].item())**2)

    
    

    


    for idx_box in range(boxes.shape[0]):
        if idx_box==id_box or tmp_bools[idx_box]==0:
            continue
            
        tmp_box=boxes[idx_box]
        h_tmp_box=np.sqrt((tmp_box[1].item()-tmp_box[3].item())**2)

        if ratio_a_in_b(box,tmp_box)>0.90 and np.sqrt((h_tmp_box-h_box)**2)<30:
            tmp_bools[idx_box]=0
            tmp_bools[id_box]=0


    

    return boxes,scores,class_labels,tmp_bools

def suppress_blank_boxes(id_box,boxes,scores,class_labels,img,tmp_bools):
    box=boxes[id_box]
    
    if not 0 in img[int(box[3].item()):int(box[1].item()),int(box[0].item()):int(box[2].item())]:
        tmp_bools[id_box]=0
        return boxes,scores,class_labels,tmp_bools

    return boxes,scores,class_labels,tmp_bools

def last_correction_sec_head_left_side(id_box,boxes,scores,class_labels,img):
    if class_labels[id_box]!=7:
        return boxes

    if boxes[id_box,0]<=500:
        boxes[id_box,0]=20
    return boxes

def last_correction_text_captions(id_box,boxes,scores,class_labels,img,tmp_bools):
    if class_labels[id_box]!=0 or tmp_bools[id_box]==0:
        return boxes,scores,class_labels,tmp_bools
        
    box=boxes[id_box]
    
    min_dist=None
    min_label=None
    min_idx=None


    for idx_box in range(boxes.shape[0]):
        if idx_box==id_box or tmp_bools[idx_box]==0:
            continue
            
        tmp_box=boxes[idx_box]
        
        dist=np.sqrt((box[1].item()-tmp_box[3].item())**2)
        if (min_dist==None or dist<min_dist) and box[2]>=tmp_box[0]:
            min_dist=dist
            min_label=class_labels[idx_box]
            min_idx=idx_box

        if ratio_a_in_b(box,tmp_box)>0.95 and class_labels[idx_box] in [6,8]:
            min_label=class_labels[idx_box]
            min_idx=idx_box
            break

        

    if not min_label in [6,8]:
        tmp_bools[id_box]=0
    else:
        tmp_txt_eros=extract_words_bb(box,img,show=False,erosion=True)
        tmp_txt=extract_words_bb(box,img,show=False,erosion=False)
        if len(tmp_txt_eros)>len(tmp_txt):
            tmp_txt=tmp_txt_eros

        if "figure" in tmp_txt.lower()[:10]:
            class_labels[min_idx]=6

        if "table" in tmp_txt.lower()[:10]:
            class_labels[min_idx]=8


   

    return boxes,scores,class_labels,tmp_bools


def last_correction_picture_bb(id_box,boxes,scores,class_labels,img,tmp_bools):
    if boxes[id_box]!=6:
        return boxes,scores,class_labels,tmp_bools

    #picture is not centered



            

        

        
        
        