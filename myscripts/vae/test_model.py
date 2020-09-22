"""Test trained model with test data"""

from torch.utils.data.dataloader import DataLoader
from myscripts.vae.trainer import VAEDatasets
from torch.utils.data.dataset import Dataset
from myscripts.vae.vae import VAE
import torch
import trainer
from torch.utils.tensorboard import SummaryWriter
from vae_config import VAEConfiguration
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    model_path = "myscripts\\vae\\model\\train_9_15.pt"

    config = VAEConfiguration()
    device = torch.device("cuda")
    model = VAE(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model_name = "test_dataset"

    dataset = VAEDatasets(config, ToTensor())
    loader = DataLoader(dataset, **config.loader_args)

    writer = SummaryWriter(f"{config.LOG_DIR}_{model_name}")

    trainer.test(1, config, model, device, loader, writer)
