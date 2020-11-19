"""Configuration for rendering with path tracing and VAE BSSRDF"""

variant = "gpu_rgb"
rr_depth = 5
max_depth = 3
spp = 1
sample_per_pass = 1

film_width = 4
film_height = 4

seed = 4

model_name = "best_model"

enable_bssrdf = True
visualize_invalid_sample = True
multi_process = False

