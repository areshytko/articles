#!/usr/bin/env python3
"""

"""

import os
import pathlib
from typing import Any, Dict, List

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# pylint: disable=no-member


class SimpleMnistExample(pl.LightningModule):
    """
    Simple pytorch-lightning system: FFNN on MNIST dataset to
    show pytorch-lightning distributed learning capabilities.
    """

    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

        self.train_dataset = None
        self.valid_dataset = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def prepare_data(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_dir = pathlib.Path.home() / 'data'
        train_set = datasets.MNIST(data_dir, train=True,
                                   download=True, transform=transform)
        valid_set = datasets.MNIST(data_dir, train=False,
                                   download=True, transform=transform)

        self.train_dataset = train_set
        self.valid_dataset = valid_set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=64)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=64)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self,
                      batch: Dict[str, Any],
                      batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        predict = self(x)
        loss = F.nll_loss(predict, y)

        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self,
                        batch: Dict[str, Any],
                        batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        predict = self(x)
        loss = F.nll_loss(predict, y)

        predict = torch.argmax(predict, dim=1)
        correct = (predict == y).sum()

        return {'val_loss': loss, 'correct': correct, 'size': y.numel()}

    def validation_epoch_end(self,
                             outputs: List[Any]) -> Dict[str, Any]:

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        correct = torch.stack([x['correct'] for x in outputs]).sum().float()
        overall = torch.FloatTensor([x['size'] for x in outputs]).sum()
        val_accuracy = correct / overall

        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_acc': val_accuracy
        }

        print(f"\n\nValidation epoch results:\n{tensorboard_logs}\n")

        return {'avg_val_loss': avg_loss, 'val_acc': val_accuracy, 'log': tensorboard_logs}


@hydra.main(config_path='ddp_train_example.yaml')
def main(conf: OmegaConf):

    if 'RANK' in os.environ:
        os.environ['NODE_RANK'] = os.environ['RANK']

    model = SimpleMnistExample()

    trainer = pl.Trainer(gpus=conf.gpus,
                         num_nodes=conf.num_nodes,
                         distributed_backend=conf.distributed_backend,
                         max_epochs=conf.max_epochs)
    trainer.fit(model)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
