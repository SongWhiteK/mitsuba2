import numpy as np


from traindata_config import TrainDataConfiguration



class ParamGenerator:
    """
    Sample and compute medium parameters
    """
    # Range:
    #   eta, U(1, 1.5)
    #   effective_albedo, U(0, 1)
    #   g, U(0, 0.95)
    # Ref:
    #   scale, 10000
    #   sigma_t, 0.264
    #   albedo, 1
    #   isotropy

    def get_eta(self):
        return 1 + np.random.rand()*0.5

    def get_g(self):
        return np.random.rand()*0.95

    def get_albedo(self):
        return np.random.rand()

    def get_sigmat(self):
        return 1.

    def sample_params(self):
        """
        Sample medium parameters
        fractive index, anisotrpic parameter, effective albedo, sigmat
        """
        medium = {}
        medium["eta"] = self.get_eta()
        medium["g"] = self.get_g()
        medium["albedo"] = self.get_albedo()
        medium["sigma_t"] = self.get_sigmat()

        if TrainDataConfiguration().DEBUG:
            print(medium)

        return medium

class FixedParamGenerator(ParamGenerator):
    """
    Fixed medium parameters generator
    """

    def get_albedo(self):
        return 0.9

    def get_eta(self):
        return 1.5

    def get_g(self):
        return 0.0


def get_reduced_albedo(albedo, g, sigmat):
    sigmas = albedo * sigmat
    sigmaa = sigmat - sigmas
    return (1 - g) * sigmas / ((1 - g) * sigmas + sigmaa)

def get_reduced_sigmat(albedo, g, sigmat):
    sigmas = albedo * sigmat
    sigmaa = sigmat - sigmas
    return (1 - g) * sigmas + sigmaa

def effective_albedo_2_reduced_albedo(effective_albedo):
    return (1 - np.exp(-8 * effective_albedo)) / (1 - np.exp(-8))

def reduced_albedo_to_effective_albedo(reduced_albedo):
    return -np.log(1.0 - reduced_albedo * (1.0 - np.exp(-8.0))) / 8.0

def get_sigman(medium):
    """
    Compute standard deviation of scattering in a medium
    Cite from D. Vicini [2019]
    """
    albedo = medium["albedo"]
    g = medium["g"]
    sigmat = medium["sigma_t"]
    
    reduced_albedo = get_reduced_albedo(albedo, g, sigmat)
    reduced_sigmat = get_reduced_sigmat(albedo, g, sigmat)
    effective_albedo = reduced_albedo_to_effective_albedo(reduced_albedo)

    MAD = 0.25 * (g + reduced_albedo) + effective_albedo

    return 2*MAD / reduced_sigmat


def scale_mat_2_str(mat):
    mat_str = "{x} 0 0 0 0 {y} 0 0 0 0 {z} 0 0 0 0 1".format(
        x=mat[0, 0], y=mat[1, 1], z=mat[2, 2])

    return mat_str


def get_d_in(res):
    theta, phi = np.meshgrid(
        np.linspace(0, np.pi/3, res),
        np.linspace(0, 2*np.pi, 6*res)
    )
    theta = np.ravel(theta)
    phi = np.ravel(phi)

    wi = sph_dir(theta, phi)

    return wi


def sph_dir(theta, phi):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    result = np.zeros([len(theta), 3])
    result[:, 0] = cp * st
    result[:, 1] = sp * st
    result[:, 2] = ct
    return result



