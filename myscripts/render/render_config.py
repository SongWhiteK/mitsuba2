"""Configuration for rendering with path tracing and VAE BSSRDF"""

from os import truncate


variant = "gpu_rgb"
rr_depth = 5
max_depth = 5
spp = 256
sample_per_pass = 8192

film_width = 512
film_height = 512

seed = 4

model_name = "best_model"
im_size = 63

enable_bssrdf = True
visualize_invalid_sample = True
multi_process = False
aovs = True

zoom = True

scale = 1