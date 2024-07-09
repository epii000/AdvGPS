from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
import torch
from opencood.utils import box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from collections import OrderedDict
from opencood.utils.transformation_utils import x1_to_x2, dist_two_pose
# import  opencood.data_utils.datasets.COM_RANGE as com_range
import numpy as np

import pdb

import multiprocessing

class Attack_Process():


    def __init__(self,hypes,train=True):
        self.hypes = hypes
        self.pre_processor = build_preprocessor(hypes['preprocess'],
                                                train)
        print('Attack_Process init!')
    
    def merge_features_to_dict(self,processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict
    

    def process(self,base_data_dict,processered_lidar_np,transformation_matrix):

        ego_lidar_pose = []
        self.transformation_matrix = transformation_matrix

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        processed_lidar_list = []
        processed_features = []
        index = 0
        for i, (cav_id, selected_cav_base) in enumerate(base_data_dict.items()):

            # print("5################")

            # check if the cav is within the communication range with ego
            distance = dist_two_pose(selected_cav_base['params']['lidar_pose'],
                                     ego_lidar_pose)

            if distance > 70:
                continue



            # lidar_np = selected_cav_base['lidar_np']
            # lidar_np = shuffle_points(lidar_np)
            # lidar_np = mask_ego_points(lidar_np)
            
            lidar_np = processered_lidar_np[0][index]
            lidar_np = torch.tensor(lidar_np)
            
            lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                transformation_matrix[0,index])
            
            # print("baolu:",self.hypes['preprocess'][
            #                                 'cav_lidar_range'])
            lidar_np = mask_points_by_range(lidar_np,
                                        self.hypes['preprocess'][
                                            'cav_lidar_range'])

            ###grad  detach
            # print("AA################")
            processed_feature = self.pre_processor.preprocess(lidar_np)


            # print("processed_feature:###############", type(processed_feature))
            processed_features.append(processed_feature)
            index = index + 1

        merged_feature_dict = self.merge_features_to_dict(processed_features)
        processed_lidar_list.append(merged_feature_dict)
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)

        return processed_lidar_torch_dict
    
    def process_pose(self,base_data_dict,processered_lidar_np,pose):

        transformation_matrix = []
        for index in range(pose.shape[0]):
            v1 = x1_to_x2(pose[index][0],pose[index][1])
            transformation_matrix.append(v1)
        #transformation_matrix = torch.tensor(transformation_matrix)

        ego_lidar_pose = []
        

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        processed_lidar_list = []
        processed_features = []
        projected_lidar_stack = []
        index = 0
        for i, (cav_id, selected_cav_base) in enumerate(base_data_dict.items()):

            # print("5################")

            # check if the cav is within the communication range with ego
            distance = dist_two_pose(selected_cav_base['params']['lidar_pose'],
                                     ego_lidar_pose)

            if distance > 70:
                continue



            # lidar_np = selected_cav_base['lidar_np']
            # lidar_np = shuffle_points(lidar_np)
            # lidar_np = mask_ego_points(lidar_np)
            
            lidar_np = processered_lidar_np[0][index]
            lidar_np = torch.tensor(lidar_np)
            
            lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                transformation_matrix[index])
            
            # print("baolu:",self.hypes['preprocess'][
            #                                 'cav_lidar_range'])
            lidar_np = mask_points_by_range(lidar_np,
                                        self.hypes['preprocess'][
                                            'cav_lidar_range'])

            projected_lidar = lidar_np.detach().numpy()
            projected_lidar = \
                    np.r_[projected_lidar.T,
                          [np.ones(projected_lidar.shape[0])*i]].T
            projected_lidar_stack.append(projected_lidar)

            ###grad  detach
            # print("AA################")
            processed_feature = self.pre_processor.preprocess(lidar_np)

            #v2 = self.pre_processor.preprocess(lidar_np.numpy())

            # if processed_feature != v2:
            #     print('unsame!!!')


            # print("processed_feature:###############", type(processed_feature))
            processed_features.append(processed_feature)
            index = index + 1

        vis_lidar = np.vstack(projected_lidar_stack)
        vis_lidar = torch.from_numpy(vis_lidar)

        merged_feature_dict = self.merge_features_to_dict(processed_features)
        processed_lidar_list.append(merged_feature_dict)
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)

        return processed_lidar_torch_dict,vis_lidar

    def processed_feature_pool(self,input):

        selected_cav_base = input[1]
        i = input[0]

        # pdb.set_trace()
        print("!!!!!!!!!!!!!!!!!", i)

        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np = torch.tensor(lidar_np)
        lidar_np[:, :3] = \
        box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                            self.transformation_matrix[0,i])
        
        lidar_np = mask_points_by_range(lidar_np,
                                    self.hypes['preprocess'][
                                        'cav_lidar_range'])

        ###grad  detach
        print("!!!!!!!!!!!!!!!!!11111", i)
        processed_feature = self.pre_processor.preprocess(lidar_np)
        print("!!!!!!!!!!!!!!!!!22222", i)


        return  processed_feature
    
    def multi_process(self,base_data_dict,transformation_matrix):

        self.ego_lidar_pose = []
        self.transformation_matrix = transformation_matrix

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                self.ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(self.ego_lidar_pose) > 0

        processed_lidar_list = []
        # processed_features = []
        output = []

        # pool = multiprocessing.Pool()

        # 创建进程列表
        processes = []
        for i, (cav_id, selected_cav_base) in enumerate(base_data_dict.items()):

            print("start")
            output=[i, selected_cav_base]
            process = multiprocessing.Process(target=self.processed_feature_pool, args=(output,))
            process.start()
            processes.append(process)

        # 等待进程完成
        for process in processes:
            process.join()
        # print("start")
        # pdb.set_trace()
        # processed_features = pool.map(self.processed_feature_pool, output)
        # pool.close()
        # pool.join()

        print("done!")


        merged_feature_dict = self.merge_features_to_dict(processed_features)
        processed_lidar_list.append(merged_feature_dict)
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)

        return processed_lidar_torch_dict

 

    