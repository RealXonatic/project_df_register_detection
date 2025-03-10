

import matplotlib.pyplot as plt
import pdfplumber
from pdf2image import convert_from_path
from detectron2.structures import Boxes, Instances
import pytesseract
import json 
from PIL import Image

import torchmetrics
import argparse
from torchmetrics.detection import MeanAveragePrecision
import cv2

from pprint import pprint

from ditod import add_vit_config
from tqdm import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from ditod.VGTTrainer import DefaultPredictor
import cv2 as cv
import matplotlib.image as mpimg
from transformers import AutoTokenizer


import sys
sys.path.append("./../")

from post_processings import*
from pre_processings import*
from dataset_utils import*


from PIL import ImageFont, ImageDraw
from PIL import Image



font = ImageFont.truetype("./post_processings/DejaVuSans.ttf", 50)


colors={"Caption": (208, 156, 250,125), #violet
        "Footnote": (0, 255, 0,125),
        "Formula": (0, 0, 255,125),
        "List-item": (255, 255, 0,125),
        "Page-footer": (0, 255, 255,125),
        "Page-header": (255, 0, 0,125), 
        "Picture": (255, 211, 113,125),#jaune
        "Section-header": (0, 141, 218,125), #bleu 
        "Table": (84, 180, 53,125), #vert 
        "Text": (255,244,85,125),#(65, 201, 226,125), #bleu plus clair 
        "Title": (175, 130, 96,125)}




# Function to viz the annotation
def markup(c,image,page, boxes,scores,class_labels,drawing_utils,pdf_name=""):
    ''' Draws the segmentation, bounding box, and label of each annotation
    '''
    page_id=page.page_number

    model,L_header,coord_line,L_footer=drawing_utils

    image=np.array(image)

    if model!=None:
        #image[model.numpy()==0,:]=np.array([255,255,255])
    
        if L_footer!=None:
            for k in L_footer:
                pass
                #image[k,160:image.shape[1]-200,:]=np.array([0,0,0])
    
        if L_header!=None and coord_line!=None:
            for k in L_header:
                if coord_line<1000:
                    pass
                    #image[k,int(coord_line):image.shape[1]-200,:]=np.array([0,0,0])
                else:
                    pass
                    #image[k,200:int(coord_line),:]=np.array([0,0,0])
    

    
    
    image = Image.fromarray(np.uint8(image))

    
                

    draw = ImageDraw.Draw(image, 'RGBA')
    for box_id in range(boxes.shape[0]):
        # Draw segmentation
        xa=min(boxes[box_id][0].item(),boxes[box_id][2].item())
        ya=min(boxes[box_id][1].item(),boxes[box_id][3].item())
        xb=max(boxes[box_id][0].item(),boxes[box_id][2].item())
        yb=max(boxes[box_id][1].item(),boxes[box_id][3].item())
    
        draw.rectangle([(xa,ya),(xb,yb)],fill=colors[id_to_label[class_labels[box_id].item()]])

    for box_id in range(boxes.shape[0]):
         
        # Draw label
        xa=min(boxes[box_id][0].item(),boxes[box_id][2].item())
        ya=min(boxes[box_id][1].item(),boxes[box_id][3].item())
        xb=max(boxes[box_id][0].item(),boxes[box_id][2].item())
        yb=max(boxes[box_id][1].item(),boxes[box_id][3].item())

        h=np.sqrt((ya-yb)**2)
        w=np.sqrt((xa-xb)**2)

        if h<100:
            ya+=h

        tmp_text=id_to_label[class_labels[box_id].item()]+" "+str(np.round(scores[box_id].item()*100,0))+"%"
       
        tbb_xa,tbb_ya,tbb_xb,tbb_yb = draw.textbbox(xy=(xa,ya),text=tmp_text,font=font)
        w=np.sqrt((tbb_xa-tbb_xb)**2)*1.25
        h=np.sqrt((tbb_ya-tbb_yb)**2)*1.13
        draw.rectangle(
            (xa,
             ya,
             xa + w,
             ya + h),
            fill=(64, 64, 64, 255)
        )
        
        draw.text(
        (xa,
         ya),
        text=tmp_text,
        fill=(255, 255, 255, 255),
        font=font
        )
        
    fig=plt.figure(figsize=(30,21))
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.savefig("./results/"+pdf_name+str(page_id)+".pdf",format="pdf",bbox_inches='tight')
    plt.show()
    
    cv2.imwrite("./results/"+pdf_name[:-4]+"_page_"+str(int(page.page_number))+"_post.png",cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    
    print("ok")
   
    #plt.close('all')



def run_VGT_single_p(page,pdf_name,cfg,predictor,id_to_label,traitement=None,plot=False,save=False):
    #--image_root 'pages/' 
    image_root='pages/' 
    #--grid_root 'pages/' 
    grid_root='pages/' 
    page_id=page.page_number
    
    image_path=image_root+pdf_name[:-4]+"_page_"+str(int(page.page_number))+".png"
    grid_path = grid_root + pdf_name[:-4]+"_page_"+str(int(page.page_number))+ ".pdf.pkl"

    
    # Step 5: run inference
    img = cv2.imread(image_path,cv2.COLOR_BGR2RGB)
        
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


  
    if save:
        cv2.imwrite("./results/"+pdf_name[:-4]+"_page_"+str(int(page.page_number))+".png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


    img_BW=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_BW=np.array(img_BW.copy())
    img_BW[img_BW<200]=0
    img_BW[img_BW>=200]=255

    
    
    


    #nexperia
    cond_1= (img[:,:,0]==123) & (img[:,:,1]==111) & (img[:,:,2]==3)
    cond_2= (img[:,:,0]==122) & (img[:,:,1]==110) & (img[:,:,2]==3)

    #vishay
    cond_3= (img[:,:,0]==243) & (img[:,:,1]==229) & (img[:,:,2]==204)
    img[cond_1]=torch.FloatTensor([0,0,0])
    img[cond_2]=torch.FloatTensor([0,0,0])
    img[cond_3]=torch.FloatTensor([255,255,255])


    txt=pytesseract.image_to_string(img_BW[:500,:], lang='eng')

    if "barry industries" in txt.lower():
        return False,None,None,None,None

   

    

    

 
                
                

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=["Caption","Footnote","Formula","List-item","Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"])

    output = predictor(img, grid_path)["instances"]
   


    boxes = correct_box_orientation(output.pred_boxes.tensor.detach().cpu())
    scores = output.scores.detach().cpu()
    class_labels = output.pred_classes.detach().cpu()

    status=True
    drawing_utils=(None,None,None,None)
    if not traitement is None:

        status,boxes,scores,class_labels,drawing_utils=traitement(boxes,scores,class_labels,page.page_number,img)

    
    if plot and status:
        height, width = img.shape[:2]

        new_instances = Instances((height, width), 
                              pred_boxes=Boxes(boxes.clone()), 
                              scores=scores, 
                              pred_classes=class_labels.int())
    
        # Use the Visualizer to draw the filtered predictions
        v = Visualizer(img[:, :, ::-1], 
                       metadata=md, 
                       scale=1.0, 
                       instance_mode=ColorMode.SEGMENTATION)  # Adjust color mode as needed
    
        #result = v.draw_instance_predictions(new_instances.to("cpu"))
        
        #result_image = result.get_image()[:, :, ::-1]

        #plt.figure(figsize=(20,10))
        #plt.imshow(result_image)
        img_PIL = Image.fromarray(img.copy())
        markup(id_to_label,img_PIL,page,boxes,scores,class_labels.int(),drawing_utils,pdf_name )

    return status,boxes,scores,class_labels,img 

def get_VGT(config,WEIGHTS):
    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config)
    #step2: options
    cfg.merge_from_list(['MODEL.WEIGHTS', WEIGHTS])
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    
    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    return predictor