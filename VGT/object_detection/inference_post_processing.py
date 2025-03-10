import argparse

import cv2

from ditod import add_vit_config

from utils import *  #fonction pour le post-processing
import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from ditod.VGTTrainer import DefaultPredictor

import json
import os

from post_processings import *

def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--image_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--image_name",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_root",
        help="Name of the output visualization file.",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.dataset in ('D4LA', 'doclaynet'):
        image_path = args.image_root + args.image_name + ".png"
    else:
        image_path = args.image_root + args.image_name + ".jpg"

    if args.dataset == 'publaynet':
        grid_path = args.grid_root + args.image_name + ".pdf.pkl"
    elif args.dataset == 'docbank':
        grid_path = args.grid_root + args.image_name + ".pkl"
    elif args.dataset == 'D4LA':
        grid_path = args.grid_root + args.image_name + ".pkl"
    elif args.dataset == 'doclaynet':
        grid_path = args.grid_root + args.image_name + ".pdf.pkl"

    output_file_name = args.output_root + args.image_name + ".jpg"

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)

    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: run inference
    img = cv2.imread(image_path)

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if args.dataset == 'publaynet':
        md.set(thing_classes=["text","title","list","table","figure"])
    elif args.dataset == 'docbank':
        md.set(thing_classes=["abstract","author","caption","date","equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"])
    elif args.dataset == 'D4LA':
        md.set(thing_classes=["DocTitle","ParaTitle","ParaText","ListText","RegionTitle", "Date", "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText", "RegionKV", "Regionlist", "Abstract", "Author", "TableName", "Table", "Figure", "FigureName", "Equation", "Reference", "Footnote", "PageHeader", "PageFooter", "Number", "Catalog", "PageNumber"])
    elif args.dataset == 'doclaynet':
        md.set(thing_classes=["Caption","Footnote","Formula","List-item","Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"])

    output = predictor(img, grid_path)["instances"]

    # Visualisation des résultats avant post-processing
    v = Visualizer(img[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    cv2.imwrite(output_file_name, result_image)
    print(f"Image avec prédictions enregistrée dans {output_file_name}")

    # Récupérer les infos de boxes, score et classes
    boxes = correct_box_orientation(output.pred_boxes.tensor.cpu())
    scores = output.scores.detach().cpu()
    class_labels = output.pred_classes.detach().cpu()
    
    # Définition du chemin de sortie pour les résultats JSON
    json_post_processed_path = os.path.join(args.output_root, args.image_name + "_results.json")

    # Vérification que `md` contient `thing_classes`
    if hasattr(md, 'thing_classes'):
        post_processed_results = [{"class": md.thing_classes[int(class_labels[i])],
                                   "score": float(scores[i]),
                                   "bbox": boxes[i].tolist()} for i in range(len(boxes))]
    else:
        raise AttributeError("L'objet `md` ne contient pas `thing_classes`. Vérifiez sa définition.")

    # Sauvegarde des résultats post-traités au format JSON
    with open(json_post_processed_path, "w") as f:
        json.dump(post_processed_results, f, indent=4)

    print(f"✅ Résultats avant post-processing enregistrés dans {json_post_processed_path}")
    
    
    # Post-processing

    # === Étape 1 : Appliquer le post-processing ===
    _,f_b,f_s,f_l,_ = traitement_boxes_lw(boxes, scores, class_labels, page_id=1, img=img)
    
    # === Étape 2 : Sauvegarder les résultats post-traités ===
    json_post_processed_path = os.path.join(args.output_root, args.image_name + "_postprocessed_results.json")
    
    if hasattr(md, 'thing_classes'):
        post_processed_results = [
            {"class": md.thing_classes[int(f_l[i])], "score": float(f_s[i]), "bbox": f_b[i].tolist()}
            for i in range(len(f_b))
        ]
    else:
        raise AttributeError("L'objet `md` ne contient pas `thing_classes`. Vérifiez sa définition.")
    
    with open(json_post_processed_path, "w") as f:
        json.dump(post_processed_results, f, indent=4)
    
    print(f"✅ Résultats après post-processing enregistrés dans {json_post_processed_path}")
    
    # === Étape 3 : Visualisation des résultats post-traités ===
    v_post = Visualizer(img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                        md, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    result_post = v_post.draw_instance_predictions(output.to("cpu"))
    result_post_image = result_post.get_image()[:, :, ::-1]
    
    # === Étape 4 : Sauvegarder l'image post-traitée ===
    processed_image_path = os.path.join(args.output_root, args.image_name + "_processed.jpg")
    cv2.imwrite(processed_image_path, result_post_image)
    print(f"✅ Image après post-processing enregistrée dans {processed_image_path}")

if __name__ == '__main__':
    main()