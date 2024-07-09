"""
Convert lidar to voxel
"""
import sys

import numpy as np
import torch

import pdb

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor
import opencood.utils.common_utils as common_utils

class VoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(VoxelPreprocessor, self).__init__(preprocess_params, train)
        # TODO: add intermediate lidar range later
        self.lidar_range = self.params['cav_lidar_range'] #-140.8,-38.4,-5,140.8,38.4,3

        # self.vw = self.params['args']['vw']
        # self.vh = self.params['args']['vh']
        # self.vd = self.params['args']['vd']
        # self.T = self.params['args']['T']
        self.vw = self.params['args']['voxel_size'][0] # 0.4
        self.vh = self.params['args']['voxel_size'][1] # 0.4
        self.vd = self.params['args']['voxel_size'][2] # 8
        self.T = self.params['args']['max_points_per_voxel'] # 32

        # pdb.set_trace()
        self.lidar_range = self.params['cav_lidar_range']   #-140.8,-38.4,-5,140.8,38.4,3
        self.voxel_size = self.params['args']['voxel_size'] #0.4,0.4,8


        #TODO:jinlong

        self.vw = torch.tensor(self.vw)
        self.vh = torch.tensor(self.vh)
        self.vd = torch.tensor(self.vd)
        self.T = torch.tensor(self.T)


        #TODO:baolu
        grid_size = (torch.tensor(self.lidar_range[3:6]) -
                     torch.tensor(self.lidar_range[0:3])) / torch.tensor(self.voxel_size) # 704.0, 191.99999999999997, 1.0
        self.grid_size = torch.round(grid_size).type(torch.int64) # 704, 192, 1

    def preprocess(self, pcd_np):
        """
        Preprocess the lidar points by  voxelization.

        Parameters
        ----------
        pcd_np : np.ndarray
            The raw lidar.

        Returns
        -------
        data_dict : the structured output dictionary.
        """
        data_dict = {}

        # calculate the voxel coordinates
        # voxel_coords = ((pcd_np[:, :3] -
        #                  np.floor(np.array([self.lidar_range[0],
        #                                     self.lidar_range[1],
        #                                     self.lidar_range[2]])) / (
        #                      self.vw, self.vh, self.vd))).astype(np.int32)
        #TODO:jinlong
        # pcd_np = torch.tensor(pcd_np)

        if len(pcd_np) == 0:
            data_dict['voxel_features'] = []
            data_dict['voxel_coords'] = []
            data_dict['voxel_num_points'] = []
            # pdb.set_trace()

            return data_dict
        _, is_numpy = common_utils.check_numpy_to_torch(pcd_np)
        if is_numpy:
            device = 'cpu'
            pcd_np = torch.tensor(pcd_np)
            pcd_np = pcd_np.to(device)
            grid_size = self.grid_size.to(device)

            #TODO:baolu
            voxel_coords = torch.floor((pcd_np[:, :3] -
                         torch.tensor([self.lidar_range[0],
                                            self.lidar_range[1],
                                            self.lidar_range[2]]).to(device) ) / torch.tensor(
                             [self.vw, self.vh, self.vd]).to(device))
            


            ### TODO:baolu
            non_index = torch.where(voxel_coords>=grid_size)
            non_index_2 = torch.where(voxel_coords < 0)
            none_index = torch.cat((non_index[0],non_index_2[0]))
            if len(none_index) != 0:
                for index in none_index:

                    x1 = voxel_coords[0:index,:]
                    x2 = voxel_coords[index+1:,:]
                    voxel_coords = torch.cat((x1,x2),dim=0)
                    x3 = pcd_np[0:index,:]
                    x4 = pcd_np[index+1:,:]
                    pcd_np = torch.cat((x3,x4),dim=0)

            voxel_coords = voxel_coords[:, [2, 1, 0]]
            
            

            # 获取唯一值和索引
            voxel_coords, inv_ind = torch.unique(voxel_coords, sorted=False, return_inverse=True, dim=0)

            # 获取计数
            _, voxel_counts = torch.unique(inv_ind, return_counts=True)


            voxel_features = torch.zeros((len(voxel_coords), self.T, 4),dtype=torch.float32).to(device)
            for i in range(len(voxel_coords)):
                pts = pcd_np[inv_ind == i]
                if len(pts) > self.T:
                    pts = pts[:self.T]
                    voxel_counts[i] = self.T
                voxel_features[i, :pts.shape[0],:] = pts

            data_dict['voxel_features'] = voxel_features
            data_dict['voxel_coords'] = voxel_coords
            data_dict['voxel_num_points'] = voxel_counts
            return data_dict




            # voxel_coords = np.floor((pcd_np[:, :3] -
            #              np.array([self.lidar_range[0],
            #                                 self.lidar_range[1],
            #                                 self.lidar_range[2]])) / np.array(
            #                  [self.vw, self.vh, self.vd]))

           
            # voxel_coords = voxel_coords[:, [2, 1, 0]]
            # voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
            #                                                 return_inverse=True,
            #                                                 return_counts=True)

            
            
            # voxel_features = np.zeros((len(voxel_coords), self.T, 4),dtype=np.float32)
            # for i in range(len(voxel_coords)):
            #     pts = pcd_np[inv_ind == i]
            #     if len(pts) > self.T:
            #         pts = pts[:self.T]
            #         voxel_counts[i] = self.T
            #     voxel_features[i, :pts.shape[0],:] = pts


            # data_dict['voxel_features'] = torch.from_numpy(np.array(voxel_features))
            # data_dict['voxel_coords'] = torch.from_numpy(voxel_coords)
            # data_dict['voxel_num_points'] = torch.from_numpy(voxel_counts)
            # return data_dict
        else:


            device = 'cuda'
            pcd_np = pcd_np.to(device)
            grid_size = self.grid_size.to(device)

            #TODO:baolu
            voxel_coords = torch.floor((pcd_np[:, :3] -
                         torch.tensor([self.lidar_range[0],
                                            self.lidar_range[1],
                                            self.lidar_range[2]]).to(device) ) / torch.tensor(
                             [self.vw, self.vh, self.vd]).to(device))
            


            ### TODO:baolu
            non_index = torch.where(voxel_coords>=grid_size)
            non_index_2 = torch.where(voxel_coords < 0)
            none_index = torch.cat((non_index[0],non_index_2[0]))
            if len(none_index) != 0:
                for index in none_index:

                    x1 = voxel_coords[0:index,:]
                    x2 = voxel_coords[index+1:,:]
                    voxel_coords = torch.cat((x1,x2),dim=0)
                    x3 = pcd_np[0:index,:]
                    x4 = pcd_np[index+1:,:]
                    pcd_np = torch.cat((x3,x4),dim=0)

            voxel_coords = voxel_coords[:, [2, 1, 0]]
            
            

            # 获取唯一值和索引
            voxel_coords, inv_ind = torch.unique(voxel_coords, sorted=False, return_inverse=True, dim=0)

            # 获取计数
            _, voxel_counts = torch.unique(inv_ind, return_counts=True)


            voxel_features = torch.zeros((len(voxel_coords), self.T, 4),dtype=torch.float32).to(device)
            for i in range(len(voxel_coords)):
                pts = pcd_np[inv_ind == i]
                if len(pts) > self.T:
                    pts = pts[:self.T]
                    voxel_counts[i] = self.T
                voxel_features[i, :pts.shape[0],:] = pts


            # for i in range(len(voxel_coords)):
            #     if i == 0:
            #         voxel = torch.zeros((self.T, 4), dtype=torch.float32).to(device)#######torch.float32

            #         pts = pcd_np[inv_ind == i]
            #         if voxel_counts[i] > self.T:
            #             pts = pts[:self.T, :]
            #             voxel_counts[i] = self.T

            #         voxel[:pts.shape[0], :] = pts
            #         voxel_features = voxel.unsqueeze(0)
            #     else:

            #         voxel = torch.zeros((self.T, 4), dtype=torch.float32).to(device)######torch.float32
            #         pts = pcd_np[inv_ind == i]
            #         if voxel_counts[i] > self.T:
            #             pts = pts[:self.T, :]
            #             voxel_counts[i] = self.T

            #         voxel[:pts.shape[0], :] = pts
            #         voxel_features = torch.cat((voxel_features,voxel.unsqueeze(0)),dim=0)

            data_dict['voxel_features'] = voxel_features.cpu()
            data_dict['voxel_coords'] = voxel_coords.cpu()
            data_dict['voxel_num_points'] = voxel_counts.cpu()
            return data_dict






    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords}

    @staticmethod
    def collate_batch_dict(batch: dict):

        if len(batch['voxel_features']) == 0:
            return {
            'voxel_features': batch['voxel_features'],
            'voxel_coords': batch['voxel_coords'],
            'voxel_num_points': batch['voxel_num_points']
        }
        
        voxel_features = torch.cat([feature for feature in batch['voxel_features']])
        voxel_num_points = torch.cat([num_points for num_points in batch['voxel_num_points']])

        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i].detach().numpy(), ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))


        return {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points
        }

