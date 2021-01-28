"""Getting data for paper"""

import sys
sys.path.append("./myscripts/gen_train/")
sys.path.append("./myscripts/render/")
sys.path.append("./myscripts/vae/")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from vae import VAE
from vae_config import VAEConfiguration
import utils
import utils_render
from PIL import Image

import plots

# Prepare outging position and height map before running this script

def main():
    n_sample = 5000
    save = True

    # Model setting
    config = VAEConfiguration()
    device = torch.device("cuda")
    model = VAE(config).to(device)

    model_path = "./myscripts/vae/model/best_model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Medium parameter setting
    sigma_t = 1.0
    albedo = 0.9
    g = 0.5
    eta = 1.5
    medium = {"albedo":albedo, "g":g, "sigma_t":sigma_t, "eta":eta}

    sigma_n = utils.get_sigman(medium)
    eff_albedo = utils.reduced_albedo_to_effective_albedo(utils.get_reduced_albedo(albedo, g, sigma_t))
    medium["sigma_n"] = sigma_n
    medium["eff_albedo"] = eff_albedo
    print(f"sigma n: {sigma_n}")
    sys.exit()


    # In plane case
    path_plane = "./myscripts/train_data/virtical/sample00.csv"
    pos_pt, pos_vae = plots.plots_plane(n_sample, model, device, medium, path_plane, save)


    # Angled incident plot
    path_angle = ["./myscripts/train_data/virtical/sample00.csv",
                  "./myscripts/train_data/x_225/sample00.csv",
                  "./myscripts/train_data/x_450/sample00.csv",
                  "./myscripts/train_data/x_675/sample00.csv"]
    pos_pt_angle, pos_vae_angle = plots.plots_angle(n_sample, model, device, medium, path_angle, save)

    # Plot CDF
    p_cdf = p_cdf = np.linspace(0, 1, n_sample)
    plots.com_cdf(p_cdf, pos_pt, pos_pt_angle, pos_vae, pos_vae_angle, medium["sigma_n"], save)


    # Evaluate vae for embossed shape and output as csv for blender
    path_map = "./myscripts/data_process/emboss_map.png"
    plots.save_emboss(n_sample, path_map, model, device, medium)

    # Absorption plot
    path_abs = "./myscripts/train_data/abs_sample/sample00.csv"
    plots.plots_abs(model, device, medium, path_abs, save)


if __name__ == '__main__':
    main()
