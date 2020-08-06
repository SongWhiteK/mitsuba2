import numpy as np



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
    def __init__(self, seed=4):
        np.random.seed(seed)

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

        return medium

class FixedParamGenerator(ParamGenerator):
    """
    Fixed medium parameters generator
    """

    def __init__(self, seed=4):
        super().__init__(seed)

    def get_albedo(self):
        return 0.99

    def get_eta(self):
        return 1.5

    def get_g(self):
        return 0.5


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






        




