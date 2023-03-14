import numpy as np
import pandas as pd
import mmcv
import PIL
from streamlit_agraph import agraph,Node,Edge,Config
from streamlit_agraph import Config as aconfig
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from streamlit_option_menu import option_menu

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged', 'background'
]

PREDICATES = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]



class Model:
    def __init__(self,model_ckt,cfg,device='cpu'):
        #model_ckt ='/home/agens/conda_user/aivg/streamlit/models/epoch_60.pth'
        #cfg = Config.fromfile('/home/agens/conda_user/aivg/streamlit/models/psgtr_r50_psg_inference.py')
        self.model = init_detector(cfg, model_ckt, device=device)

    def image_extraction(self,image,num_rel=20):
                 
        
        img = image  # (H, W, 3)
        img_h, img_w = img.shape[:-1]
        
        # Decrease contrast
        img = PIL.Image.fromarray(img)
        converter = PIL.ImageEnhance.Color(img)
        img = converter.enhance(0.01)
    #    if out_file is not None:
    #       mmcv.imwrite(np.asarray(img), 'bw'+out_file)

        # Draw masks
        result = inference_detector(self.model, image)
        pan_results = result.pan_results

        ids = np.unique(pan_results)[::-1]
        num_classes = 133
        legal_indices = (ids != num_classes)  # for VOID label
        ids = ids[legal_indices]

        # Get predicted labels
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        labels = [CLASSES[l] for l in labels]

        # Object Labels
        rel_obj_labels = result.labels
        rel_obj_labels = [CLASSES[l - 1] for l in rel_obj_labels]
        n_rel_topk = min(num_rel, len(result.labels)//2)
        rel_dists = result.rel_dists[:, 1:]
        rel_scores = rel_dists.max(1)
        rel_sort_idx = np.argsort( rel_scores)[::-1]
        rel_labels_topk = rel_dists[rel_sort_idx].argmax(1)
        rel_pair_idxes_topk = result.rel_pair_idxes[rel_sort_idx]
        relations = np.concatenate(
        [rel_pair_idxes_topk, rel_labels_topk[..., None],rel_scores[rel_sort_idx,None]], axis=1)
        n_rels = len(relations)
        if n_rels < num_rel:
            num_rel = n_rels
        all_rel = []
        for i, r in enumerate(relations):
            s_idx, o_idx, rel_id, pos_value = r
            s_idx , o_idx, rel_id = s_idx.astype(np.int64),o_idx.astype(np.int64),rel_id.astype(np.int64)
            s_label = rel_obj_labels[s_idx]
            o_label = rel_obj_labels[o_idx]
            rel_label = PREDICATES[rel_id]
            all_rel.append([s_label,rel_label,o_label,pos_value])
            if i == num_rel:
                break;

        return pd.DataFrame(all_rel, columns =['SUBJECT','PREDICATE','OBJECT','VALUE'] )



    def graph_vis(self,df,src,dst,rel):

        config = aconfig(width=500, 
                height=500,
             #   graphviz_layout=layout,
                directed=True,
                nodeHighlightBehavior=True, 
                highlightColor="#F7A7A6",
                collapsible=True,
                node={'labelProperty':'label'},
                maxZoom=2,
                minZoom=0.1,
                staticGraphWithDragAndDrop=False,
                staticGraph=False,
                initialZoom=1
                ) 

        nodes= set(df[src].values)|set(df[dst].values)
        nodes = [Node(id=i, label=i,color="#DBEBC2") for i in nodes]
        edges = [Edge(source=i, target=k,label=j,color="#F7A7A6",arrows_to=True) for i,j,k in zip(df[src],df[rel],df[dst])]
        return agraph(nodes,edges,config)

