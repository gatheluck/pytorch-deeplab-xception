import torch
from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, weight_bb, output_stride, BatchNorm):
	if backbone == 'resnet':
		model = resnet.ResNet101(output_stride, BatchNorm)
		model._load_pretrained_model(weight_bb)
		#model.load_state_dict(torch.load(weight_bb))
		return model
	elif backbone == 'xception':
		return xception.AlignedXception(output_stride, BatchNorm)
	elif backbone == 'drn':
		return drn.drn_d_54(BatchNorm)
	elif backbone == 'mobilenet':
		return mobilenet.MobileNetV2(output_stride, BatchNorm)
	else:
		raise NotImplementedError
