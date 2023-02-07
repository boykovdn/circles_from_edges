import os
import torch
import scipy
import imageio
import numpy as np
import matplotlib.pyplot as plt

import pykeops 
pykeops.set_bin_folder("./pykeops-1.5-cpython-38/")

from scipy.ndimage.morphology import distance_transform_edt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm

from torch_kde import GaussianKDE, ParabolicKDE
from Unet import SimpleUnet
from utils import get_fourier_from_mask
from transforms import rescale_to

class ContourFitter():

    def __init__(self, DSET_PATH, 
            DSET_FILENAMES, 
            N_SUBSAMPLES, 
            N_LATENT_COMPONENTS, 
            LATENT_SIGMA,
            PROBAB_SIGMA,
            EDGE_MODEL_STATE,
            LR,
            DEVICE=0, 
            DEBUG_N=None):

        self.DSET_PATH = DSET_PATH
        self.DSET_FILENAMES = DSET_FILENAMES
        self.N_SUBSAMPLES = N_SUBSAMPLES
        self.N_LATENT_COMPONENTS = N_LATENT_COMPONENTS
        self.LATENT_SIGMA = LATENT_SIGMA
        self.PROBAB_SIGMA = PROBAB_SIGMA
        self.EDGE_MODEL_STATE = EDGE_MODEL_STATE
        self.LR = LR
        self.DEVICE = DEVICE
        self.DEBUG_N = DEBUG_N

        self.components_, self.mean_, self.all_coefs = self.get_latent_pca(
                DSET_PATH, 
                DSET_FILENAMES, 
                N_SUBSAMPLES, 
                N_LATENT_COMPONENTS, 
                DEVICE=DEVICE, 
                DEBUG_N=DEBUG_N, 
                return_all_coefs=True)

        mean_reshape = self.mean_.unsqueeze(-1).expand(-1,self.all_coefs.shape[-1])
        self.pca_dset = (self.all_coefs - mean_reshape).T @ self.components_.T

        # Keops KDEs:
        self.latent_kde = ParabolicKDE(self.pca_dset, weights=torch.Tensor((1/len(self.all_coefs),)).expand(DEBUG_N).to(DEVICE),sigma=LATENT_SIGMA)
        self.edge_model = self.get_edge_model(EDGE_MODEL_STATE, DEVICE=DEVICE)

    def optimize(self, img, centroid_, initial_state=None, OPTIM_STEPS=150):
        r"""
        Inputs:
            :img: torch.Tensor raw input to the edge model. It is turned into 
                the edge probability map.
        """

        # TODO Allow for edge map to be reused, so that you can fit more efficiently
        # multiple contours in the same frame without calculating the map again and again.
        probab_kde = self.get_edge_probab_kde(self.edge_model, img, self.PROBAB_SIGMA, DEVICE=self.DEVICE)

        if initial_state is None:
            # Initialize to the mean shape
            # TODO Mode makes more sense, but harder to do.
            initial_state = torch.zeros(1, self.N_LATENT_COMPONENTS, device=self.DEVICE)
            initial_state.requires_grad = True

        optimizer = torch.optim.Adam([initial_state], lr=self.LR)
        final_state = self.run_latent_grad_descent(
                initial_state, 
                probab_kde, 
                self.latent_kde, 
                centroid_, 
                self.components_, 
                self.mean_, 
                optimizer, 
                OPTIM_STEPS, 
                self.N_SUBSAMPLES, 
                DEVICE=self.DEVICE)

        return final_state

    def get_latent_pca(self, DSET_PATH, DSET_FILENAMES, N_SUBSAMPLES, N_LATENT_COMPONENTS, DEVICE=0, DEBUG_N=None, return_all_coefs=False):
    
        if DEBUG_N is None:
            DEBUG_N = len(DSET_FILENAMES)
    
        all_coefs = np.zeros((2*N_SUBSAMPLES, len(DSET_FILENAMES[:DEBUG_N])))
    
        for idx, filename in enumerate(tqdm(DSET_FILENAMES[:DEBUG_N], desc="PCA/ Collecting fourier coefs")):
            img_ = imageio.imread("{}/{}".format(DSET_PATH, filename))
            coef, freq_ = get_fourier_from_mask(img_, n_subsamples=N_SUBSAMPLES)
            all_coefs[:N_SUBSAMPLES, idx] = coef.real
            all_coefs[N_SUBSAMPLES:, idx] = coef.imag
    
        print("PCA/ Fitting latent representation")
        std_clf = make_pipeline(StandardScaler(with_std=False), PCA(n_components=N_LATENT_COMPONENTS))
        pca_dset = std_clf.fit_transform(all_coefs.T)
    
        mean_ = torch.from_numpy(std_clf[0].mean_).float().to(DEVICE)
        components_ = torch.from_numpy(std_clf[1].components_).float().to(DEVICE)
    
        if return_all_coefs:
            return components_, mean_, torch.from_numpy(all_coefs).float().to(DEVICE)
        else:
            return components_, mean_
    
    def get_edge_model(self, EDGE_MODEL_STATE, DEVICE=0):
        edge_model = SimpleUnet().to(DEVICE)
        edge_model_state = torch.load(EDGE_MODEL_STATE)
        edge_model.load_state_dict(edge_model_state)
        print("Loaded edge model: {}".format(EDGE_MODEL_STATE))
        return edge_model
    
    def get_edge_probab_kde(self, edge_model, img_, PROBAB_SIGMA, DEVICE=0):
        with torch.no_grad():
            edge_probabs = edge_model(rescale_to(torch.from_numpy(img_).unsqueeze(0).unsqueeze(0).to(DEVICE), to=(0,1))).detach().sigmoid()
            import pdb; pdb.set_trace()
            H,W = edge_probabs[0,0].shape
            # Here we use apply kde on the image grid, with the weight of each pixel
            # equal to its probability of being part of the edge (predicted by edge model).
            probab_coords = torch.from_numpy(np.mgrid[0:H, 0:W].reshape(2,H*W)).float()
            probab_kde = GaussianKDE(probab_coords.T.to(DEVICE), weights=edge_probabs[0].reshape(H*W)/(H*W), sigma=PROBAB_SIGMA)
    
            return probab_kde

    def latent_state_to_coords(self, latent_init, centroid_, components_, mean_, N_SUBSAMPLES, DEVICE=0):
        fourier_coefs = (latent_init @ components_) + mean_
        ### fourier to image space ###
        reals = fourier_coefs[..., :N_SUBSAMPLES]
        imags = fourier_coefs[..., N_SUBSAMPLES:]
        contour_1d = torch.fft.ifft(reals + imags*1j).real # (100,50)
        ### Distances to contour coords ###
        hs = (centroid_[0] - contour_1d * (torch.arange(N_SUBSAMPLES, device=DEVICE) * ((2*np.pi)/N_SUBSAMPLES)).cos()).relu() # (100,50), latent samples, image space points
        ws = (centroid_[1] + contour_1d * (torch.arange(N_SUBSAMPLES, device=DEVICE) * ((2*np.pi)/N_SUBSAMPLES)).sin()).relu()
        ### Coords to probabilities ###
        contour_coords = torch.stack([hs,ws], dim=-1).to(DEVICE)

        return contour_coords

    def run_latent_grad_descent(self, latent_init, probab_kde, latent_kde, centroid_, components_, mean_, optimizer, OPTIM_STEPS, N_SUBSAMPLES, DEVICE=0):

        for it_op_ in tqdm(range(OPTIM_STEPS)):

            contour_coords = self.latent_state_to_coords(latent_init,
                    centroid_,
                    components_,
                    mean_,
                    N_SUBSAMPLES, DEVICE=DEVICE)

            contour_edge_probabs = probab_kde(contour_coords.reshape(N_SUBSAMPLES, 2))
    
            latent_loss = latent_kde(latent_init.T).mean()
            probab_loss = -contour_edge_probabs.mean() * 100000
            loss = probab_loss + latent_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        return latent_init

def main():

    DSET_PATH = "/mnt/fast0/biv20/datasets/temp/mask"
    DSET_FILENAMES = os.listdir(DSET_PATH)
    BACKGROUND_ID = 0
    N_SUBSAMPLES = 50
    N_LATENT_COMPONENTS = 50
    PROBAB_SIGMA = 20
    LATENT_SIGMA = 300
    OPTIM_STEPS = 150
    LR = 50
    LR_C = 0
    EDGE_MODEL_STATE = "/mnt/fast0/biv20/experiments/exp14_unet_edgedet/model_unet.state" # TODO
    DEVICE = 0
    INIT_CIRC_R = 50

    DEBUG_N = 1000

    cf = ContourFitter(DSET_PATH,
            DSET_FILENAMES,
            N_SUBSAMPLES,
            N_LATENT_COMPONENTS,
            LATENT_SIGMA,
            PROBAB_SIGMA,
            EDGE_MODEL_STATE,
            LR,
            DEVICE=0,
            DEBUG_N=500)

    # TODO 09.08.2021: Runs, but haven't checked if correct results
    img_ = imageio.imread("/mnt/fast0/biv20/datasets/eggression/eggression_0.png")
    centroid_ = torch.Tensor([50.,50.]).float()
    final_state = cf.optimize(img_, centroid_)

if __name__ == "__main__":
    main()
