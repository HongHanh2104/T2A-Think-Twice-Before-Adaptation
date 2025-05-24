import torch
import torch.nn as nn

from adapters.base_adapter import BaseAdapter
from losses import compute_noise_tolerant_negative_loss
from utils import compute_cosine_similarity


class T2AAdapter(BaseAdapter):
    def setup_adapter(self):
        self.steps = self.config.get("steps")
        self.episodic = self.config.get("episodic")
        self.noise_type = self.config.get("noise_type")
        self.gamma = self.config.get("gamma")
        self.l1_lambda = self.config.get("l1_lambda")
        self.psi = self.config.get("psi")
        self.alpha = self.config.get("alpha")
        self.beta = self.config.get("beta")

    def setup_model(self):
        """Configure model for adaptation"""
        # Setup optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.setup_optimizer(params)
        # Store states for reset
        self.model_state = self.model.state_dict()
        self.optimizer_state = self.optimizer.state_dict()

    def reset(self):
        """Reset model and optimizer to initial state"""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

    @torch.enable_grad()
    def entropy_minimization(self, data_dict: dict):
        logits = self.model(data_dict)["cls"]
        entropy = self.entropy_fn(logits)
        coeff = 1 / (torch.exp(entropy - self.e_margin))
        loss = (entropy * coeff).mean(0)
        return loss

    @torch.enable_grad()
    def noise_tolerant_negative_loss(self, data_dict: dict):
        outputs = self.model(data_dict)["cls"]
        loss = compute_noise_tolerant_negative_loss(
            outputs, noise_type=self.noise_type, gamma=self.gamma, alpha=self.alpha, beta=self.beta
        )
        return loss

    def perform_gradient_masking(self):
        bn_grads = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.weight.grad is not None:
                    bn_grads.append(module.weight.grad.flatten())
                if module.bias.grad is not None:
                    bn_grads.append(module.bias.grad.flatten())

        bn_grad_vector = torch.cat(bn_grads)
        for name, param in self.model.named_parameters():
            if param.grad is not None and not any(
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                for n, m in self.model.named_modules()
                if name in n
            ):
                param_grad_flat = param.grad.flatten().unsqueeze(0)
                cos_sim = compute_cosine_similarity(
                    param_grad_flat, bn_grad_vector, strategy=self.cosine_strategy
                )
                if cos_sim < self.psi:
                    param.grad.zero_()

    @torch.enable_grad()
    def forward(self, data_dict: dict) -> dict:
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            loss = self.entropy_minimization(
                data_dict
            ) + self.noise_tolerant_negative_loss(data_dict)
            for param in self.model.parameters():
                if param.requires_grad:
                    loss += self.l1_lambda * torch.norm(param, p=1)
            self.optimizer.zero_grad()
            loss.backward()
            if self.filter_grad:
                self.perform_gradient_masking()
            self.optimizer.step()

        with torch.no_grad():
            logits = self.model(data_dict)["cls"]
            prob = logits.softmax(1)[:, 1]

        return {"cls": logits, "prob": prob}
