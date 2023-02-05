from torch.utils.data import Dataset
import os
import open3d as o3d
import torch
#from trimesh.sample import sample_surface
from scipy.spatial import cKDTree
from tqdm import tqdm
import numpy as np
from dataset.utils import points_scale


class ReconDataset(Dataset):
    
    def __init__(self, data_path, point_batch):
        super().__init__

        self.point_batch = point_batch
        model = np.genfromtxt(data_path)
        self.pnts = model[:,:3]
        self.normals = model[:,-3:]

        self.pnts = points_scale(self.pnts)



    def __getitem__(self, index):
        point_cloud_size = self.pnts.shape[0]

        rand_incs = np.random.choice(point_cloud_size, size=self.point_batch)

        on_surface_pnts = self.pnts[rand_incs,:]
        on_surface_normals = self.normals[rand_incs,:]

        off_surface_pnts = np.random.uniform(-1.2 , 1.2, size=(self.point_batch,3))
        off_surface_normals = np.ones((self.point_batch,3))* -1

        sdf = np.zeros((self.point_batch * 2, 1))
        sdf[self.point_batch:,:] = -1

        pnts = np.concatenate((on_surface_pnts,off_surface_pnts), axis=0)
        normals = np.concatenate((on_surface_normals,off_surface_normals), axis=0)

        return {'pnts': torch.from_numpy(pnts).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}

    def __len__(self):
        return self.pnts.shape[0] // self.point_batch



#test
#data = ReconDataset('data/hole.xyz',100)