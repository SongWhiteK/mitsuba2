"""Plot functions"""

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
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plots_plane(n_data, model, device, medium, pt_path, save):
    """Make some plots for plane situation"""

    # Get outgoing position for PT
    df = pd.read_csv(pt_path)

    x_pt = (df["p_out_x"] - df["p_in_x"])
    y_pt = (df["p_out_y"] - df["p_in_y"])
    z_pt = (df["p_out_z"] - df["p_in_z"])

    # Generate height map
    im = get_planemap(n_data)

    # Get out going position for VAE
    d = np.array([0, 0, 1])
    height_max = 0.0
    recon_pos, recon_abs = eval_vae(n_data, medium, d, height_max,
                                    im, model, device)

    x_vae = pd.Series(recon_pos[:,0] * medium["sigma_n"])
    y_vae = pd.Series(recon_pos[:,1] * medium["sigma_n"])
    z_vae = pd.Series(recon_pos[:,2] * medium["sigma_n"])

    # Plot outgoing position
    # PT
    plot_xy(x_pt, y_pt, medium["sigma_n"], "PT", save)
    # VAE
    plot_xy(x_vae, y_vae, medium["sigma_n"], "CVAE", save)

    # Output statistics
    # PT
    print("PT stats")
    print_stats(x_pt, y_pt, z_pt)

    # VAE
    print("VAE stats")
    print_stats(x_vae, y_vae, z_vae)

    save_stats(pd.DataFrame([x_pt, y_pt, z_pt]).T, pd.DataFrame([x_vae, y_vae, z_vae]).T)

    # CDF
    p_cdf = np.linspace(0, 1, n_data)

    pos_pt = [x_pt, y_pt, z_pt]
    pos_vae = [x_vae, y_vae, z_vae]

    return pos_pt, pos_vae


def plots_angle(n_data, model, device, medium, path, save):

    # Generate height map
    im = get_planemap(n_data)
    height_max = 0.0

    n_angle = len(path)
    label = ["virtical",
             r"$\theta = 22.5^{\circ}$",
             r"$\theta = 45^{\circ}$",
             r"$\theta = 67.5^{\circ}$"]

    pos_pt = [None for i in range(n_angle)]
    pos_vae = [None for i in range(n_angle)]
    mean_pt = np.zeros([n_angle, 3])
    mean_vae = np.zeros([n_angle, 3])

    for i in range(n_angle):
        # PT data
        df_pt = pd.read_csv(path[i])
        pos_pt[i] = [df_pt["p_out_x"] - df_pt["p_in_x"],
                     df_pt["p_out_y"] - df_pt["p_in_y"],
                     df_pt["p_out_z"] - df_pt["p_in_z"]]
        
        #VAE data
        d = np.array([np.sin(i * np.pi / 8), 0, np.cos(i * np.pi / 8)])
        recon_pos, recon_abs = eval_vae(n_data, medium, d, height_max,
                                        im, model, device)

        pos_vae[i] = [pd.Series(recon_pos[:,0] * medium["sigma_n"]),
                      pd.Series(recon_pos[:,1] * medium["sigma_n"]),
                      pd.Series(recon_pos[:,2] * medium["sigma_n"])]

        # Get mean
        mean_pt[i, :] = np.array([pos_pt[i][0].mean(), pos_pt[i][1].mean(), pos_pt[i][2].mean()])
        mean_vae[i, :] = np.array([pos_vae[i][0].mean(), pos_vae[i][1].mean(), pos_vae[i][2].mean()])

    # Plot mean
    # PT
    plot_mean(n_angle, mean_pt, medium["sigma_n"], label, "PT", save)
    # VAE
    plot_mean(n_angle, mean_vae, medium["sigma_n"], label, "CVAE", save)

    # Distribution plot for 45 degree
    plot_xy(pos_pt[2][0], pos_pt[2][1], medium["sigma_n"], "PT_x45", save)
    plot_xy(pos_vae[2][0], pos_vae[2][1], medium["sigma_n"], "CVAE_x45", save)

    # CDF for 45 degree
    p_cdf = np.linspace(0, 1, n_data)

    return pos_pt[2], pos_vae[2]
    

def save_emboss(n_data, path_map, model, device, medium):

    # Generate height map
    im = get_embossmap(n_data, path_map)

    d = np.array([0, 0, 1])
    height_max = 1.0
    recon_pos, recon_abs = eval_vae(n_data, medium, d, height_max,
                                    im, model, device)

    df = pd.DataFrame()
    df["x"] = pd.Series(recon_pos[:,0] * medium["sigma_n"])
    df["y"] = pd.Series(recon_pos[:,1] * medium["sigma_n"])
    df["z"] = pd.Series(recon_pos[:,2] * medium["sigma_n"] + 20.7995)

    df.to_csv("./myscripts/data_process/emboss_out.csv")


def com_cdf(p_cdf, pos_pt, pos_pt_angle, pos_vae, pos_vae_angle, sigma_n, save):

    var = ["x", "y", "z"]

    for i in range(3):
        plt.figure(figsize=(7, 6))
        plt.title("CDF for " + var[i], fontsize=20)
        plt.grid()
        plt.xlim(-10 * sigma_n, 10 * sigma_n)
        plt.ylim(0, 1.1)
        plt.xlabel(f"{var[i]} [mm]", fontsize=20)
        plt.ylabel("cdf", fontsize=20)
        plt.plot(pos_pt[i].sort_values(), p_cdf, label=f"{var[0]}_pt", linewidth=4)
        plt.plot(pos_vae[i].sort_values(), p_cdf, label=f"{var[0]}_cvae", linewidth=4, linestyle="--")
        plt.legend(fontsize=15)

        if save:
            plt.savefig(f"./myscripts/data_process/out/cdf_{var[i]}.png")
        else:
            plt.show()

        plt.figure(figsize=(7, 6))
        plt.title("CDF for " + var[i] + " (Angled Incident)", fontsize=20)
        plt.grid()
        plt.xlim(-10 * sigma_n, 10 * sigma_n)
        plt.ylim(0, 1.1)
        plt.xlabel(f"{var[i]} [mm]", fontsize=20)
        plt.ylabel("cdf", fontsize=20)
        plt.plot(pos_pt_angle[i].sort_values(), p_cdf, label=f"{var[0]}_pt_angled", linewidth=4)
        plt.plot(pos_vae_angle[i].sort_values(), p_cdf, label=f"{var[0]}_cvae_angled", linewidth=4, linestyle="--")
        plt.legend(fontsize=15)
        
        if save:
            plt.savefig(f"./myscripts/data_process/out/cdf_{var[i]}_angled.png")
        else:
            plt.show()


def plots_abs(model, device, medium, path, save):

    # Resolution of 2d plot
    res = 30
    n_data = res * res * 6 

    # Get absorption probability
    # PT
    df = pd.read_csv(path)
    abs_pt = df["abs_prob"].values

    # VAE
    # Generate height map
    im = get_planemap(n_data)
    height_max = 0.0

    d = utils.get_d_in(res)
    recon_pos, recon_abs = eval_vae_sphere(n_data, medium, d, height_max,
                                    im, model, device)

    abs_vae = np.array(recon_abs)

    abs_min = np.min([np.min(abs_pt), np.min(abs_vae)])
    abs_max = np.max([np.max(abs_pt), np.max(abs_vae)])

    # Plot PT
    map_abs(abs_pt, res, abs_min, abs_max, "PT", save)
    # Plot VAE
    map_abs(abs_vae, res, abs_min, abs_max, "CVAE", save)


def plot_xy(x, y, sigma_n, data_type, save):
    plt.figure(figsize=(8, 8), tight_layout=True)
    plt.grid()
    plt.xlim(-10 * sigma_n, 10 * sigma_n)
    plt.ylim(-10 * sigma_n, 10 * sigma_n)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Outgoing position for " + data_type, fontsize=20)
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.scatter(x, y)
    if save:
        plt.savefig(f"./myscripts/data_process/out/plane_dist_{data_type}.png")
    else:
        plt.show()

def print_stats(x, y, z):
    print(f"mean x: {x.mean()}")
    print(f"mean y: {y.mean()}")
    print(f"mean z: {z.mean()}")
    print(f"std x: {x.std()}")
    print(f"std y: {y.std()}")
    print(f"std z: {z.std()}")


def save_stats(pt, vae):
    mat = np.array([[pt.mean().values, pt.std().values],
                    [vae.mean().values, vae.std().values]])
    mat = np.reshape(mat, [2,6])

    df = pd.DataFrame(mat, index=["PT", "VAE"], columns=["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"])

    df.to_csv("./myscripts/data_process/out/stats.csv")


def get_planemap(n_data):

    canvas = np.ones([63, 63], dtype="uint8") * 63

    uv_range = np.arange(1, 64)
    V, U = np.meshgrid(uv_range, uv_range)
    mask = (V - 32)**2 + (U - 32)**2 <= 31**2
    canvas[np.logical_not(mask)] = 0
    canvas_tensor = torch.tensor(canvas)

    im = torch.ones([n_data, 1, 63, 63])

    for i in range(n_data):
        im[i, 0, :, :] = canvas_tensor

    return im

def get_embossmap(n_data, path_map):
    canvas = Image.open(path_map)
    canvas = np.array(canvas)

    uv_range = np.arange(1, 64)
    V, U = np.meshgrid(uv_range, uv_range)
    mask = (V - 32)**2 + (U - 32)**2 <= 31**2
    canvas[np.logical_not(mask)] = 0

    canvas_tensor = torch.tensor(canvas)

    im = torch.ones([n_data, 1, 63, 63])
    for i in range(n_data):
        im[i, 0, :, :] = canvas_tensor

    return im


def eval_vae(n_data, medium, d, height_max, im, model, device):

    d = d / np.linalg.norm(d)
    props = torch.ones([n_data,7]) * torch.tensor([medium["eff_albedo"], medium["g"],
                                                  medium["eta"], d[0], d[1], d[2], height_max])

    # model evalation
    with torch.no_grad():
        feature = model.feature_conversion(im.to(device, dtype=torch.float), props.to(device, dtype=torch.float))

        latent_z = torch.randn(n_data, 4).to(device, dtype=torch.float)

        recon_pos, recon_abs = model.decode(feature, latent_z)

        recon_pos = recon_pos.cpu()
        recon_abs = recon_abs.cpu()

    return recon_pos, recon_abs

def eval_vae_sphere(n_data, medium, d, height_max, im, model, device):

    props = torch.ones([n_data,7]) * torch.tensor([medium["eff_albedo"], medium["g"],
                                                   medium["eta"], 0, 0, 0, height_max])
    props[:, 3:-1] = torch.tensor(d)

    # model evalation
    with torch.no_grad():
        feature = model.feature_conversion(im.to(device, dtype=torch.float), props.to(device, dtype=torch.float))

        latent_z = torch.randn(n_data, 4).to(device, dtype=torch.float)

        recon_pos, recon_abs = model.decode(feature, latent_z)

        recon_pos = recon_pos.cpu()
        recon_abs = recon_abs.cpu()

    return recon_pos, recon_abs
    

def plot_mean(n_angle, mean, sigma_n, label, data_type, save):

    plt.figure(figsize=(8, 8), tight_layout=True)
    plt.grid()
    plt.xlim(-sigma_n, sigma_n)
    plt.ylim(-sigma_n, sigma_n)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Mean of outgoing position for " + data_type, fontsize=20)
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    
    for i in range(n_angle):
        plt.scatter(mean[i, 0], mean[i, 1], label=label[i], linewidths=10)

    plt.legend(fontsize=20)

    if save:
        plt.savefig(f"./myscripts/data_process/out/mean_{data_type}.png")
    else:
        plt.show()


def map_abs(abs_prob, res, abs_min, abs_max, data_type, save):
    values_r = abs_prob.reshape(6 * res, res).T

    # Plot values for spherical coordinates
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    im_plot = ax.imshow(values_r, extent=[0, 2 * np.pi, np.pi/3, 0], vmin=abs_min,
                                          vmax=abs_max, cmap='jet', interpolation='bicubic')

    plt.title("Absorption probability of " + data_type, fontsize=15)
    ax.set_xlabel(r'$\phi$', size=14)
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', '$\\pi$', '$2\\pi$'], fontsize=15)
    ax.set_ylabel(r'$\theta$', size=14)
    ax.set_yticks([0, np.pi/3])
    ax.set_yticklabels(['0', '$\\pi/3$'], fontsize=15)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im_plot, cax=cax)

    if save:
        plt.savefig(f"./myscripts/data_process/out/abs_{data_type}.png")
    else:
        plt.show()
    
