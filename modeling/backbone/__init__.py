import torch
from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, bb_weight, output_stride, BatchNorm):
	if backbone == 'resnet101':
		model = resnet.ResNet101(output_stride, BatchNorm)
		model._load_pretrained_model(bb_weight)
		#model.load_state_dict(torch.load(bb_weight))
		return model
	elif backbone == 'resnet50':
		model = resnet.ResNet50(output_stride, BatchNorm)
		model._load_pretrained_model(bb_weight)
		return model
	elif backbone == 'xception':
		return xception.AlignedXception(output_stride, BatchNorm)
	elif backbone == 'drn':
		return drn.drn_d_54(BatchNorm)
	elif backbone == 'mobilenet':
		return mobilenet.MobileNetV2(output_stride, BatchNorm)
	else:
		raise NotImplementedError
