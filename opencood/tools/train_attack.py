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
    parser.add_argument('--model', default='/home/jinlongli/2.model_saved/3.Attack_V2V2023/attfuse_fintune_voxel_2023_07_21_07/net_epoch29.pth',
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
                             num_workers=0,
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
    # epoches = hypes['train_params']['epoches']
    epoches=1
    # used to help schedule learning rate


    ###baolu
    attack_process = Attack_Process(hypes,train=False)


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
        # num_iter=40
        # step_size = 0.001 #TODO:baolu 0.001 - 0.01   0.02


        num_iter=hypes['attack']['num_iter_per']
        step_size = hypes['attack']['step_size']

        if not os.path.exists(os.path.join(saved_path,'npy')):
            os.makedirs(os.path.join(saved_path,'npy'))
        if not os.path.exists(os.path.join(saved_path,'vis')):
            os.makedirs(os.path.join(saved_path,'vis'))
        
        for i, batch_data in enumerate(data_loader):
            # if i < 207:
            #     continue
             
            globals.flag = True
            globals.flag_num = 0
            print("i", i)

            loss_start = 0
            loss_after_attack = 0
            for inter in range(num_iter):
                start_time = time.time()

                ###### load the same data 
                # batch_data = opencood_validate_dataset.__getitem__(150)

                optimizer.zero_grad()
                if globals.flag:#### the first project

                    #pdb.set_trace()
                    transformation_matrix = torch.tensor(np.array(batch_data['ego']['transformation_matrix'])).type(torch.float32)

                    #TODO:baolu
                    
                    point_cloud_cav = batch_data['ego']['base_data_dict'][0]
                    processered_lidar_np = batch_data['ego']['processered_lidar_np']
                    # max_T = transformation_matrix.clone()*1.05
                    # min_T = transformation_matrix.clone()*0.95
                    # print("max_T ", max_T)
                    # print("min_T ", min_T)
                    #transformation_matrix[0,0,3] = torch.tensor([10,10,10,20],dtype=torch.float)
                    globals.T_learnable = transformation_matrix
                    
                    #globals.T_learnable = transformation_matrix*0.99

                    # print(globals.flag_num, "  flag:  ", globals.flag, "  the first  T: ", transformation_matrix)

                else:####### the rest 

                    # transformation_matrix = globals.T_learnable.cpu().detach().clone()
                    transformation_matrix = globals.T_learnable
                    globals.flag_num = globals.flag_num + 1

                    # transformation_matrix = torch.where(transformation_matrix > max_T, max_T,transformation_matrix)
                    # transformation_matrix = torch.where(transformation_matrix< min_T, min_T, transformation_matrix)
                    # print(globals.flag_num, " flag:  ", globals.flag, "  transformation_matrix: ", transformation_matrix)


                T_learnable = globals.T_learnable.requires_grad_()
                #print(T_learnable)
             
                processed_feature = attack_process.process(point_cloud_cav,processered_lidar_np,T_learnable)
                # processed_feature = attack_process.multi_process(point_cloud_cav,T_learnable)
                batch_data['ego']['processed_lidar'] = processed_feature


                if inter == 0:
                    batch_data['ego'].pop('base_data_dict')
                    batch_data['ego'].pop('processered_lidar_np')
                    batch_data['ego'].pop('transformation_matrix')
                batch_data = train_utils.to_device(batch_data, device)
                
                # optimizer.zero_grad()
                # with torch.no_grad():
                
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

                
                #######update the gradient of T learnable
                loss_attack  = final_loss
                loss_attack.backward()

                if globals.flag:
                    loss_start = loss_attack.item()
                loss_after_attack = loss_attack.item()
                grad_T_learnable = T_learnable.grad 
                # print('i:',  globals.flag_num, "   ################   before_T_learnable", T_learnable)

                T_learnable = T_learnable + step_size*grad_T_learnable.sign()

                # print('i:',  globals.flag_num, "   ################   After_T_learnable", T_learnable)

                #T_learnable.grad.data.zero_() #梯度值清零
                globals.T_learnable = T_learnable.detach()
                globals.flag = False
                end_time = time.time()

                run_time = end_time - start_time
                print(num_iter, 'step_size: ', step_size, ' ',  'i:',  globals.flag_num, "one running time :  ", run_time, " s")
                

                    

            loss_attack_add =  loss_after_attack - loss_start
            txt_log.write('sample index:'+str(i)+'   loss_start:'+str(loss_start)+'   loss_after_attack:'+str(loss_after_attack)+'   attack_loss_add:'+str(loss_attack_add)+'\n')
            
                
            # criterion.logging(epoch, i, len(data_loader), writer, pbar=pbar2)
            pbar2.update(1)


            # ######add the original vkey
            
            batch_data['ego']['processed_lidar'] = processed_feature
            #batch_data['ego'].update({'transformation_matrix': gt_transformation_matrix})
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
                pred_box_tensor, pred_score, gt_box_tensor = \
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

            if True:
                npy_save_path = os.path.join(saved_path, 'npy')
                if pred_box_tensor == None:
                    continue
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)

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

    txt_log.close()
    print('Attack Testing Finished, result saved to %s' % os.path.join(opt.model,'attack'))


if __name__ == '__main__':
    main()
