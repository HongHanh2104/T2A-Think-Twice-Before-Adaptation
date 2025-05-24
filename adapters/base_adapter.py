import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class BaseAdapter(nn.Module, ABC):
    """Base class for test-time adaptation methods"""

    def __init__(self, model: nn.Module, device, **kwargs):
        super().__init__()
        self.config = kwargs
        self.device = device
        self.model = model
        self.setup_adapter()
        self.setup_model()

    @abstractmethod
    def setup_adapter(self):
        """Configure adaptation-specific parameters"""
        pass

    @abstractmethod
    def setup_model(self):
        """Configure model for adaptation"""
        pass

    def setup_optimizer(self, params):
        if self.config["optimizer"] == "Adam":
            return optim.Adam(
                params,
                lr=self.config["optimizer_config"]["lr"],
                betas=(
                    self.config["optimizer_config"]["beta1"],
                    self.config["optimizer_config"]["beta2"],
                ),
                weight_decay=self.config["optimizer_config"]["weight_decay"],
            )
        elif self.config["optimizer"] == "AdamW":
            return optim.AdamW(
                params,
                lr=self.config["optimizer_config"]["lr"],
                betas=(
                    self.config["optimizer_config"]["beta1"],
                    self.config["optimizer_config"]["beta2"],
                ),
                weight_decay=self.config["optimizer_config"]["weight_decay"],
            )
        elif self.config["optimizer"] == "SGD":
            return optim.SGD(
                params,
                lr=self.config["optimizer_config"]["lr"],
                momentum=self.config["optimizer_config"]["momentum"],
                dampening=self.config["optimizer_config"]["dampening"],
                weight_decay=self.config["optimizer_config"]["weight_decay"],
                nesterov=self.config["optimizer_config"]["nesterov"],
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.config['optimizer']} not implemented"
            )

    @torch.enable_grad()  # Enable gradients for adaptation
    def adapt_and_predict(self, data_dict: dict):
        """Adapt model and make predictions"""
        predictions = self.forward(data_dict)
        return predictions

    @abstractmethod
    def forward(self, data_dict: dict):
        """Forward pass with adaptation"""
        pass


class NoAdapter(BaseAdapter):
    """No adaptation, returns original model predictions"""

    def setup_adapter(self):
        pass

    def setup_model(self):
        self.model.eval()

    def forward(self, data_dict: dict):
        with torch.no_grad():
            return self.model(data_dict)
