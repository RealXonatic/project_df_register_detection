import torch
import matplotlib.pyplot as plt
import pytesseract
import cv2 as cv
import numpy as np

def compute_area_of_box(box):

    xmin=int(box[0].item())
    ymax=int(box[1].item())
    xmax=int(box[2].item())
    ymin=int(box[3].item())
    
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    return area

def correct_box_orientation(boxes):
    res=torch.empty(boxes.shape)
    
    for idx_box in range(boxes.shape[0]):
        x_a=boxes[idx_box][0]
        y_a=boxes[idx_box][1]
        x_b=boxes[idx_box][2]
        y_b=boxes[idx_box][3]

        min_x=min(x_a,x_b)
        min_y=min(y_a,y_b)
        max_x=max(x_a,x_b)
        max_y=max(y_a,y_b)

        res[idx_box][0]=min_x
        res[idx_box][1]=max_y
        res[idx_box][2]=max_x
        res[idx_box][3]=min_y
    return res
        
    
def compute_overlap(a, b):
    # Determine the coordinates of the intersection rectangle
    x_left = max(a[0], b[0])
    y_top = min(a[1], b[1])  # Changed to min for top
    x_right = min(a[2], b[2])
    y_bottom = max(a[3], b[3])  # Changed to max for bottom

    # Calculate the width and height of the intersection rectangle
    width = x_right - x_left
    height = y_top - y_bottom

    # If there is no overlap, return 0
    if width < 0 or height < 0:
        return 0

    # Calculate the area of the intersection rectangle
    return abs(width * height)

def ratio_a_in_b(a, b):
    area_a = abs((a[2]-a[0]) * (a[1]-a[3]))
    overlap = compute_overlap(a, b)

    if area_a == 0:
        return 0  # Prevent division by zero

    return overlap / area_a


def add_box(box,label,score,boxes,scores,class_labels):
    tmp_box=torch.zeros((1,4))
    tmp_label=torch.zeros(1)
    tmp_score=torch.zeros(1)
    
    tmp_box[0]=box.clone()
    tmp_label[0]=label
    tmp_score[0]=score

    
    
    boxes=torch.cat((boxes,tmp_box),dim=0)
    scores=torch.cat((scores,tmp_score),dim=0)
    class_labels=torch.cat((class_labels,tmp_label),dim=0)

    return boxes,scores,class_labels

def check_box_pixel(img, box):

    xmin=int(box[0].item())
    ymax=int(box[1].item())
    xmax=int(box[2].item())
    ymin=int(box[3].item())
    # Extraire la sous-zone de l'image délimitée par la box
    sub_img = img[ymin:ymax, xmin:xmax]
    # Vérifier si un pixel de valeur 0 est présent dans la sous-zone
    return np.any(sub_img == 0)

def compute_area_of_box(box):

    xmin=int(box[0].item())
    ymax=int(box[1].item())
    xmax=int(box[2].item())
    ymin=int(box[3].item())
    
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    return area


def remove_box(id_box, boxes, scores, class_labels):
    
    # Remove the box, score, and label at the specified index
    boxes = torch.cat((boxes[:id_box], boxes[id_box+1:]), dim=0)
    scores = torch.cat((scores[:id_box], scores[id_box+1:]), dim=0)
    class_labels = torch.cat((class_labels[:id_box], class_labels[id_box+1:]), dim=0)

    return boxes, scores, class_labels


def dark_background(img, box):
    
    xmin=int(box[0].item())
    ymax=int(box[1].item())
    xmax=int(box[2].item())
    ymin=int(box[3].item())
    
    # Créer des masques pour les contours extérieurs de la boîte
    top_edge = img[ymin-1:ymin, xmin:xmax+1] if ymin > 0 else np.array([])
    bottom_edge = img[ymax+1:ymax+2, xmin:xmax+1] if ymax < img.shape[0] - 1 else np.array([])
    left_edge = img[ymin:ymax+1, xmin-1:xmin] if xmin > 0 else np.array([])
    right_edge = img[ymin:ymax+1, xmax+1:xmax+2] if xmax < img.shape[1] - 1 else np.array([])
    
    # Concaténer les pixels des contours dans un seul tableau
    edge_pixels = np.concatenate((top_edge.flatten(), bottom_edge.flatten(), left_edge.flatten(), right_edge.flatten()))
    
    # Compter les pixels noirs et blancs
    black_count = np.sum(edge_pixels == 0)
    white_count = np.sum(edge_pixels == 255)
    
    # Déterminer si les pixels noirs ou blancs sont majoritaires
    if black_count > white_count:
        return 255
    else:
        return 0
   
def test_noir_x_gauche(img,x,y_min,y_max,step,background):
    return (background in img[y_min:y_max,max(0,x-step):x])


                        
def recalage_boxes_xgauche(id_box,boxes,img,step=20,attempt=False):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y

    background=dark_background(img, boxes[id_box])

    
    tmp_x=int(boxes[id_box][0].item())
    tmp_w_prec=int(boxes[id_box][2].item())-int(boxes[id_box][0].item())
    y_min=int(boxes[id_box][3].item())
    y_max=int(boxes[id_box][1].item())

    prec_x=tmp_x

    while(test_noir_x_gauche(img,tmp_x,y_min,y_max,step,background)) and tmp_x>0:
        
        tmp_x+= -step

    if not attempt:
        tmp_x+= -step
      
    boxes[id_box][0]=tmp_x

        
        
    return boxes

def test_noir_x_droit(img,x,y_min,y_max,limit,step,background):
    return (background in img[y_min:y_max,x:min(x+step,limit)])
                        
def recalage_boxes_xdroit(id_box,boxes,img,step=20,attempt=False):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y

    background=dark_background(img, boxes[id_box])

    
    tmp_x=int(boxes[id_box][2].item())
    tmp_w_prec=int(boxes[id_box][2].item())-int(boxes[id_box][0].item())
    y_min=int(boxes[id_box][3].item())
    y_max=int(boxes[id_box][1].item())

    prec_x=tmp_x

    while test_noir_x_droit(img,tmp_x,y_min,y_max,img.shape[1],step,background) and tmp_x<img.shape[1]:
        
        tmp_x+= step
        tmp_x=min(tmp_x,img.shape[1])


    if not attempt:
        tmp_x+= step
    tmp_x=min(tmp_x,img.shape[1])
    boxes[id_box][2]=tmp_x

    
        
    return boxes


def test_noir_y(img,y,x_min,x_max,step,background):

    return (background in img[max(0,y-step):y,x_min:x_max])
    
def recalage_boxes_yhaut(id_box,boxes,img,step=5,attempt=False):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y

    background=dark_background(img, boxes[id_box])

    
    tmp_y=int(boxes[id_box][3].item())
    tmp_w_prec=int(boxes[id_box][1].item())-int(boxes[id_box][3].item())
    x_min=int(boxes[id_box][0].item())
    x_max=int(boxes[id_box][2].item())

    
    

    while(test_noir_y(img,tmp_y,x_min,x_max,step,background)) and tmp_y>0:
        
        tmp_y+= -step
        tmp_y=max(tmp_y,0)
        
    
        
    if not attempt:
        tmp_y+= -step
    tmp_y=max(tmp_y,0)
    boxes[id_box][3]=tmp_y
        

    
        
    return boxes

def test_noir_ybas(img,y,x_min,x_max,limit,step,background):
    return (background in img[y:min(limit,y+step),x_min:x_max])

def recalage_boxes_ybas(id_box,boxes,img,step=5,attempt=False):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y

    background=dark_background(img, boxes[id_box])

    
    tmp_y=int(boxes[id_box][1].item())
    tmp_w_prec=int(boxes[id_box][1].item())-int(boxes[id_box][3].item())
    x_min=int(boxes[id_box][0].item())
    x_max=int(boxes[id_box][2].item())

    prec_y=tmp_y

    while(test_noir_ybas(img,tmp_y,x_min,x_max,img.shape[0],step,background)) and tmp_y<img.shape[0]:
        
        tmp_y+= step

    if not attempt:
        tmp_y+= step
    new_w=int(boxes[id_box][1].item())-tmp_y
    
   
    boxes[id_box][1]=tmp_y

    

    
        
    return boxes
    
    
    


def check_blank_space_vertical(id_box,boxes,img,min_width,mw_nb):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y
    
    box=boxes[id_box]

    starter=False
    starter_id=None
    
    for tmp_x in range(int(box[0].item()),int(box[2].item()),2):
        width_g=np.sqrt((tmp_x-int(box[0].item()))**2)
        width_d=np.sqrt((tmp_x-int(box[2].item()))**2)
        
        if (not 0 in img[int(box[3].item()):int(box[1].item()),tmp_x:tmp_x+min_width]) and (not starter) and width_g>mw_nb :
            starter=True
            starter_id=tmp_x
            
        if (0 in img[int(box[3].item()):int(box[1].item()),tmp_x:tmp_x+min_width]) and starter:
            if width_d>mw_nb:
                
                return starter_id,tmp_x
            else:
                starter=False
                starter_id=None

        

    return None,None

def check_blank_space_horizontal(id_box,boxes,img,min_width,mw_nb=150):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y
    
    box=boxes[id_box]

    starter=False
    starter_id=None
    
    for tmp_y in range(int(box[3].item()),int(box[1].item()),1):
        width_h=np.sqrt((tmp_y-int(box[3].item()))**2)
        width_b=np.sqrt((tmp_y-int(box[1].item()))**2)

        
        if (not 0 in img[tmp_y:tmp_y+min_width,int(box[0].item()):int(box[2].item())]) and (not starter) and width_h>mw_nb :
            starter=True
            starter_id=tmp_y
            
        if (0 in img[tmp_y:tmp_y+min_width,int(box[0].item()):int(box[2].item())]) and starter:
            if width_b>mw_nb:
                
                return starter_id,tmp_y
            else:
                starter=False
                starter_id=None

        

    return None,None



def divide_blank_space_vertical(id_box,boxes,scores,class_labels,img,min_width,mw_nb=500):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y

    if not class_labels[id_box] in [4,5,6]:
        return False,boxes,scores,class_labels
        
    
    
    verif=False
    while(True):
        min_x,max_x=check_blank_space_vertical(id_box,boxes,img,min_width,mw_nb)

        if min_x==None:
            break

        
        new_box=boxes[id_box].clone()
        boxes[id_box,2]=min_x
        new_box[0]=max_x
        verif=True

        boxes,scores,class_labels=add_box(new_box,class_labels[id_box],scores[id_box],boxes,scores,class_labels)
    return verif,boxes,scores,class_labels




def divide_blank_space_horizontal(id_box,boxes,scores,class_labels,img,min_width,mw_nb=30):
    #res[idx_box][0]=min_x
    #res[idx_box][1]=max_y
    #res[idx_box][2]=max_x
    #res[idx_box][3]=min_y

    
   
    verif=False
    while(True):
        min_y,max_y=check_blank_space_horizontal(id_box,boxes,img,min_width,mw_nb)

        
        if min_y==None:
            break

        
        new_box=boxes[id_box].clone()
        boxes[id_box,1]=min_y
        new_box[3]=max_y
        verif=True

        boxes,scores,class_labels=add_box(new_box,class_labels[id_box],scores[id_box],boxes,scores,class_labels)
    return verif,boxes,scores,class_labels

def supress_blank_area(id_box,boxes,img):

    box=boxes[id_box].clone()

    
    
   

    for tmp_x_gauche in range(int(box[0].item()),int(box[2].item())):
        if 0 in img[int(box[3].item()):int(box[1].item()),tmp_x_gauche:min(tmp_x_gauche+5,img.shape[1])]:
            boxes[id_box,0]=tmp_x_gauche
            break

    for tmp_x_droit in range(int(box[2].item()),int(box[0].item()),-1):
        if 0 in img[int(box[3].item()):int(box[1].item()),max(tmp_x_droit-5,0):tmp_x_droit]:
            boxes[id_box,2]=tmp_x_droit
            break

    for tmp_y_haut in range(int(box[3].item()),int(box[1].item())):
        if 0 in img[tmp_y_haut:min(tmp_y_haut+5,img.shape[0]),int(boxes[id_box,0].item()):int(boxes[id_box,2].item())]:
            boxes[id_box,3]=tmp_y_haut
            
            break
        

    for tmp_y_bas in range(int(box[1].item()),int(box[3].item()),-1):
        if 0 in img[max(tmp_y_bas-5,0):tmp_y_bas,int(boxes[id_box,0].item()):int(boxes[id_box,2].item())]:
            boxes[id_box,1]=tmp_y_bas
      
            break


        

    

            
    return boxes         

        
        
    
    

