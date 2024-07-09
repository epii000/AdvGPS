import argparse
import os
import statistics
import open3d as o3d
import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from opencood.tools import train_utils, infrence_utils
from opencood.utils import eval_utils
# import sys
# sys.path.append('/home/jinlongli/1.Detection_Set/V2V_Attack')
# sys.path.append('/home/jinlongli/1.Detection_Set/V2V_Attack/opencood')

# sys.path.append("/home/jinlong/4.3D_detection/Noise_V2V/v2vreal")
# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
# sys.path.remove("/home/jinlongli/1.Detection_Set/DA_V2V")
# print(sys.path)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

import numpy as np

import time
import random

import pdb

#global variable

from opencood.utils import globals
from opencood.data_utils.datasets.attack_process import Attack_Process
import matplotlib
matplotlib.use('Agg')




def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str,default='/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_opv2v.yaml',#required=True, #
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--model', default="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/attfuse_fintune_voxel_2023_07_21_07/net_epoch31.pth",
                        help='for fine-tuned training path')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")
    opt = parser.parse_args()
    return opt

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    print('Dataset Building')
    # opencood_train_dataset = build_dataset(hypes, visualize=False, train=True,isSim=True)#
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=True,
                                              train=False,
                                              isSim=True)
    data_loader = DataLoader(opencood_validate_dataset,
                             batch_size=1,
                             num_workers=8,
                             collate_fn=opencood_validate_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    # opencood_dataset = build_dataset(hypes, visualize=True, train=False,
    #                                  isSim=False)
    
    # train_loader = DataLoader(opencood_train_dataset,
    #                           batch_size=hypes['train_params']['batch_size'],
    #                           num_workers=0,
    #                           collate_fn=opencood_train_dataset.collate_batch_train,
    #                           shuffle=False,
    #                           pin_memory=False,
    #                           drop_last=True)

    ###baolu
    # val_loader = DataLoader(opencood_validate_dataset,
    #                         batch_size=1,
    #                         num_workers=0,
    #                         collate_fn=opencood_validate_dataset.collate_batch_test,
    #                         shuffle=False,
    #                         pin_memory=False,
    #                         drop_last=True)

    print('Loading Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#torch.device('cpu')#

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(data_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        print('Loaded model from {}'.format(saved_path))

    else:
        if opt.model:
            saved_path = train_utils.setup_train(hypes)
            model_path = opt.model
            init_epoch = 0
            #pretrained_state = torch.load(os.path.join(model_path,'latest.pth'))
            # pretrained_state = torch.load(model_path)
            # model_dict = model.state_dict()
            # pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
            # model_dict.update(pretrained_state)
            # model.load_state_dict(model_dict)
            model.load_state_dict(torch.load(model_path))
            print('Loaded pretrained model from {}'.format(model_path))
        else:
            init_epoch = 0
            # if we train the model from scratch, we need to create a folder
            # to save the model,
            saved_path = train_utils.setup_train(hypes)

    ###baolu freeze model
    for param in model.parameters():
        param.requires_grad = False

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Attack testing start')
    txt_path = os.path.join(saved_path, 'attack_eval.txt')
    txt_log = open(txt_path, "w")

    distance_path = os.path.join(saved_path, 'distance_log.txt')
    distance_log = open(distance_path, "w")


    # epoches = hypes['train_params']['epoches']
    epoches=1
    # used to help schedule learning rate


    ###baolu
    attack_process = Attack_Process(hypes,train=False)

    num_iter=hypes['Attack_params']['num_iter_per']
    step_size = [hypes['Attack_params']['pose_value'][i]/num_iter for i in range(6)]
    Is_learn =hypes['Attack_params']['pose_use'] 
    pose_limit = hypes['Attack_params']['pose_value']
    Is_attacked_pose_save = hypes['Attack_params']['attacked_pose_save']
    
    # x1_to_world = x_to_world_tensor(x1)
    # x2_to_world = x_to_world_tensor(x2)
    # world_to_x2 = torch.inverse(x2_to_world)
    # transformation_matrix = torch.mm(world_to_x2, x1_to_world)
    # x1 to x2

    from opencood.utils.transformation_utils import x_to_world_tensor,x1_to_x2
    import math
    def calc_distance(source_pose,attacked_pose,transformation_matrix_1):
        num = source_pose.shape[0]
        # pdb.set_trace()
        distance = []

        ####Transformation matrix
        pose= source_pose
        transformation_matrix = []
        for index in range(pose.shape[0]):
            v1 = x1_to_x2(pose[index][0],pose[index][1])
            transformation_matrix.append(v1)

        ####TODO:baolu  check transformation_matrix_1 and transformation_matrix , are they the same in 208th iter?

        for i in range(1,num):
            ego_pose = source_pose[i,1]
            cav_pose = source_pose[i,0]
            attacked_cav_pose = attacked_pose[i,0]
            T = torch.tensor(transformation_matrix[i]).type(torch.float32)

            cav_to_ego = T
            ego_to_cav = torch.inverse(cav_to_ego)
            ego_to_word = x_to_world_tensor(ego_pose)
            cav_to_word = x_to_world_tensor(cav_pose)
            attacked_cav_to_word = x_to_world_tensor(attacked_cav_pose)

            cav_in_ego = torch.mm(cav_to_word, ego_to_cav)
            attacked_cav_in_ego = torch.mm(attacked_cav_to_word, ego_to_cav)
            cav_location = cav_in_ego[0:3,3]
            attacked_cav_location = attacked_cav_in_ego[0:3,3]
            distance.append(math.sqrt((cav_location[0]-attacked_cav_location[0])**2 + (cav_location[1]-attacked_cav_location[1])**2 + (cav_location[2]-attacked_cav_location[2])**2))
        return distance


    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(data_loader), leave=True)
        # for i, batch_data in enumerate(train_loader):


        result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                0.7: {'tp': [], 'fp': [], 'gt': 0}}

        
        model.eval()

        
        

        if not os.path.exists(os.path.join(saved_path,'npy')):
            os.makedirs(os.path.join(saved_path,'npy'))
        if not os.path.exists(os.path.join(saved_path,'vis')):
            os.makedirs(os.path.join(saved_path,'vis'))
        if Is_attacked_pose_save:
            if not os.path.exists(os.path.join(saved_path,'attacked_pose')):
                os.makedirs(os.path.join(saved_path,'attacked_pose'))

        all_distance_dict = {}
        all_distance_list = []
        
        for batch_i, batch_data in enumerate(data_loader):



            
             
            
            print("i", batch_i)

            # if batch_i<208:

            #     continue

            loss_start = 0
            loss_after_attack = 0

            ###attack needs init!
            point_cloud_cav = batch_data['ego']['base_data_dict'][0]
            processered_lidar_np = batch_data['ego']['processered_lidar_np']
            params = batch_data['ego']['base_data_dict'][0]
            car_id = list(batch_data['ego']['base_data_dict'][0])
            pose = []
            for index in params:
                v1 = params[index]['params']['delay_cav_lidar_pose']
                v2 = params[index]['params']['cur_ego_lidar_pose']
                pose.append([v1,v2])
            pose = torch.tensor(pose).type(torch.float32)

            #learn
            add_pose = torch.zeros_like(pose)

            transformation_matrix = batch_data['ego']['transformation_matrix'][0]

            First_flag = True

            for inter in range(num_iter):
                start_time = time.time()
                optimizer.zero_grad()
            
                add_pose = add_pose.requires_grad_()
                if First_flag:


                    
                    step_size_torch = torch.zeros_like(add_pose)

                    for index in range(6):
                        if Is_learn[index]:
                            v1 = []
                            v1.append(torch.tensor([[0,0]]))
                            for i in range(1,add_pose.shape[0]):
                                if not hypes['Attack_params']['random']:####TODO:jinlong
                                    v1.append(torch.tensor([[step_size[index],0]]))

                                else:
                                    value=pose_limit[index]
                                    v1.append(torch.tensor([[random.uniform(-value,value),0]]))
                                    # v1.append(torch.tensor([[value,0]]))
                            v1 = torch.cat(v1,dim=0)
                            step_size_torch[:,:,index] = v1

                    step_size_torch = step_size_torch / 2 
            

             
                processed_feature,_ = attack_process.process_pose(point_cloud_cav,processered_lidar_np,pose+add_pose)

                if len(processed_feature['voxel_features']) == 0:
                    processed_feature = last_processed_feature
                    continue
                last_processed_feature = processed_feature
                batch_data['ego']['processed_lidar'] = processed_feature

                if inter == 0:
                    batch_data['ego'].pop('base_data_dict')
                    batch_data['ego'].pop('processered_lidar_np')
                    batch_data['ego'].pop('transformation_matrix')
                batch_data = train_utils.to_device(batch_data, device)
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

                loss_attack  = final_loss
                loss_attack.backward()

                if inter == 0:
                    loss_start = loss_attack.item()
                loss_after_attack = loss_attack.item()
                grad_T_learnable = add_pose.grad 
                

                if not hypes['Attack_params']['random']:####TODO:jinlong
                    add_pose = add_pose + step_size_torch*grad_T_learnable.sign()
                    add_pose = add_pose.detach()

                else:

                    add_pose = add_pose + step_size_torch

                ###check whether attacked pose out of limit
                for index in range(6):
                    if Is_learn[index]:
                        for i in range(1,add_pose.shape[0]):
                            if add_pose[i,0,index] > pose_limit[index]:
                                add_pose[i,0,index] = pose_limit[index]
                            if add_pose[i,0,index] < -pose_limit[index]:
                                add_pose[i,0,index] = -pose_limit[index]
                end_time = time.time()

                run_time = end_time - start_time
                print(num_iter, 'step_size: ', step_size_torch, ' ',  'i:',  inter, "one running time :  ", run_time, " s")
    
                First_flag = False



            source_pose = pose
            attacked_pose = pose + add_pose

            if Is_attacked_pose_save:
                save_data = {'car_id':car_id,'add_pose':add_pose}
                save_data_path = os.path.join(saved_path,'attacked_pose')
                save_data_full_path = os.path.join(save_data_path,str(batch_i)+'.npy')
                torch.save(save_data,save_data_full_path)

            distance = calc_distance(source_pose,attacked_pose,transformation_matrix)
            all_distance_dict[batch_i] = distance
            for item in distance:
                if item != 0:
                    all_distance_list.append(item)

            processed_feature,vis_attacked_lidar = attack_process.process_pose(point_cloud_cav,processered_lidar_np,pose+add_pose)  
            # print("the updated pose: ", T_learnable)
            loss_attack_add =  loss_after_attack - loss_start
            txt_log.write('sample index: '+str(batch_i)+'   loss_start: '+str(loss_start)+'   loss_after_attack: '+str(loss_after_attack)+'   attack_loss_add:'+str(loss_attack_add)+'\n')
            distance_log.write('sample index: '+str(batch_i)+'   distance: '+str(distance)+'\n')
            #print('sample index:'+str(i)+'   distance:'+str(distance)+'\n')   
            # criterion.logging(epoch, i, len(data_loader), writer, pbar=pbar2)
            pbar2.update(1)


            # ######add the original vkey
            
            batch_data['ego']['processed_lidar'] = processed_feature
            transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float().to(device)
            batch_data['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch})
            ################# evaluate each iteration
            # model.eval()
            
            batch_data = train_utils.to_device(batch_data, device)
            ###baolu
            with torch.no_grad():
                model.eval()
                pred_box_tensor, pred_score, gt_box_tensor,_ = \
                infrence_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_validate_dataset)
                
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)

            if hypes['Attack_params']['npy_save'] and (batch_i)%50==0:
                npy_save_path = os.path.join(saved_path, 'npy')
                
                if pred_box_tensor == None:
                    continue
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  batch_i,
                                                  vis_attacked_lidar,
                                                  npy_save_path)
                print('save the npy: ', npy_save_path)
            if False:
                vis_save_path = os.path.join(saved_path, 'vis')
                vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_validate_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  False,
                                                  vis_save_path,
                                                  dataset=opencood_validate_dataset)

            # if not os.path.exists(os.path.join(saved_path,'attack')):
            #     os.makedirs(os.path.join(saved_path,'attack'))
            
            # eval_utils.eval_final_results(result_stat,
            #                                 os.path.join(saved_path,'attack'))
        if not os.path.exists(os.path.join(saved_path,'attack')):
            os.makedirs(os.path.join(saved_path,'attack'))
        
        eval_utils.eval_final_results(result_stat,
                                os.path.join(saved_path,'attack'))
    all_distance_list = np.array(all_distance_list)
    distance_mean = all_distance_list.mean()
    distance_median = np.median(all_distance_list)
    distance_max = all_distance_list.max()
    distance_min = all_distance_list.min()
    distance_log.write('mean: '+str(distance_mean)+'  median: '+str(distance_median)+'  max: '+str(distance_max)+'  min: '+str(distance_min)+'\n')
    print('mean: '+str(distance_mean)+'  median: '+str(distance_median)+'  max: '+str(distance_max)+'  min: '+str(distance_min)+'\n')


    txt_log.close()
    distance_log.close()
    print('Attack Testing Finished, result saved to %s' % os.path.join(opt.model,'attack'))


if __name__ == '__main__':
    main()
