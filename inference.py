import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import config
import transformations
import utils
import cv2


ANCHOR_SIZES = ((4,), (8,), (16,), (32,), (64,))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)

CELL_TYPE_DICT = {'A172': 1, 'BT474': 2, 'BV2': 3, 'Huh7': 4, 'MCF7': 5, 'SHSY5Y': 6, 'SKOV3': 7, 'SkBr3': 8}

# Normalize to resnet mean and std if True.
NORMALIZE = True
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

BOX_DETECTIONS_PER_IMG = 540

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cpu')


class MaskRCNNInference():
    """
    Class for inference and evaluation of Mask R-CNN model with a ResNet-50-FPN.
    """
    def __init__(self, device=DEVICE, cell_type_dict=CELL_TYPE_DICT, checkpoint='pytorch_model-e12.bin'):
        self.checkpoint = checkpoint
        self.cell_type_dict = cell_type_dict
        self.device = device
        self.load_model()
     
    def get_model(self, num_classes, model_chkpt=None, anchor_sizes=ANCHOR_SIZES, aspect_ratios=ASPECT_RATIOS):
        
    
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
        if NORMALIZE:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                       box_detections_per_img=BOX_DETECTIONS_PER_IMG,
                                                                       rpn_anchor_generator=anchor_generator,
                                                                       image_mean=RESNET_MEAN,
                                                                       image_std=RESNET_STD)
        else:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                       box_detections_per_img=BOX_DETECTIONS_PER_IMG,
                                                                       rpn_anchor_generator=anchor_generator)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes+1)
    
        if model_chkpt:
            model.load_state_dict(torch.load(model_chkpt, map_location=self.device))
        
        return model
    
    def load_model(self):
        print("Loading:", self.checkpoint)
        self.model = self.get_model(len(self.cell_type_dict), self.checkpoint)
        # self.model.load_state_dict(torch.load(self.checkpoint), map_location=self.device)
        self.model.to(self.device)

    def preprocess_img(self, img):
        if config.resize_factor:
            img = cv2.resize(img, (int(config.WIDTH*config.resize_factor), int(config.HEIGHT*config.resize_factor)))
        
        transforms = transformations.get_transform(train=False)
        if transforms is not None: 
            img, _ = transforms(image=img, target=None)
        
        return img

    def predict(self, img, min_score=None, mask_thres=None):
        img = self.preprocess_img(img)
        self.model.eval()
        
        with torch.no_grad():
            preds = self.model([img.to(self.device)])[0]
        
#         print(preds['scores'])
        preds = utils.get_filtered_masks(preds, min_score, mask_thres)
        
        return preds
