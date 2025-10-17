from .base import BaseModel
from .cnn_model import OceanCNN
from .resnet_model import OceanResNet
from .transformer_model import OceanTransformer
from .pangu1 import OceanPangu1
from .pangu2 import OceanPangu2
from .fengwu import OceanFengwu


_model_dict = {
    "BaseModel": BaseModel,
    "OceanCNN": OceanCNN,
    "OceanResNet": OceanResNet,
    "OceanTransformer": OceanTransformer,
    "OceanPangu1": OceanPangu1,
    "OceanPangu2": OceanPangu2,
    "OceanFengwu": OceanFengwu,
}
