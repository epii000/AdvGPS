#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Jinlong Li CSU PhD
 # @Date: 2022-07-03 18:01:32
 # @LastEditors: Jinlong Li CSU PhD
 # @LastEditTime: 2023-03-31 11:57:01
### 

# source /home/jinlong/anaconda3/etc/profile.d/conda.sh


# conda activate attack



#hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_opv2v.yaml"
#hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml"
#hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_transformer.yaml"
hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_fax.yaml"
#hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_fcooper.yaml"

#model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/attfuse/net_epoch20.pth"
#model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/V2VAM/net_epoch52.pth"
#model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT/net_epoch60.pth"
#model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2x-vit/net_epoch25.pth"

#model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/attfuse_fintune_voxel_2023_07_21_07/net_epoch31.pth"
#model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/V2VAM_finetune_voxel_2023_07_21_07/net_epoch38.pth"
#model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/v2xvit_finetune_voxel_2023_07_21_10/net_epoch40.pth"
model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/CoBEVT_re_train_2023_08_18_12/net_epoch57.pth"
#model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/fcooper_re_train_2023_08_18_12/net_epoch21.pth"

#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_x_2023_08_19_21/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_y_2023_08_19_21/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_z_2023_08_19_21/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_xy_2023_08_17_14/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_xyz_2023_08_19_21/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_roll_2023_08_19_21/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_yaw_2023_08_19_21/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_pitch_2023_08_19_21/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_ryp_2023_08_19_21/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_sign_bp/1.attfuse_sign_bp_all_2023_08_19_21/attacked_pose"


#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_x_2023_08_21_10/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_y_2023_08_21_10/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_z_2023_08_21_10/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_xy_2023_08_18_14/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_xyz_2023_08_21_10/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_roll_2023_08_20_18/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_yaw_2023_08_20_18/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_pitch_2023_08_20_18/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_ryp_2023_08_20_18/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_multi_mmd/1.attfuse_multi_mmd_all_2023_08_20_18/attacked_pose"

#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_FSGM/1.attfuse_FSGM_xyz_2023_08_24_09/attacked_pose"
#attacked_pose_path="/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_FSGM/1.attfuse_FSGM_all_2023_08_24_09/attacked_pose"

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_PGD/1.attfuse_PGD_xyz_2023_08_26_11/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_PGD/1.attfuse_PGD_all_2023_08_26_11/attacked_pose'


#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_multi_mmd_shift/1.attfuse_voxelNet_multi_mmd_shift_xyz_2023_08_29_17/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_multi_mmd_shift/1.attfuse_voxelNet_multi_mmd_shift_xyz_2023_08_29_17/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_random/1.attfuse_random_xyz_2023_08_22_07/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_random/1.attfuse_random_all_2023_08_22_08/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_SOK/1.attfuse_SOK_xyz_2023_09_01_16/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_SOK/1.attfuse_SOK_all_2023_09_01_16/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_FSGM/1.attfuse_voxelNet_FSGM_xyz_2023_08_31_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_FSGM/1.attfuse_voxelNet_FSGM_all_2023_08_31_08/attacked_pose'



#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_IFSGM/1.attfuse_voxelNet_IFSGM_xyz_0.05_2023_09_02_16/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_IFSGM/1.attfuse_voxelNet_IFSGM_all_0.05_2023_09_02_16/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_PGD/1.attfuse_voxelNet_PGD_xyz_0.05_2023_09_02_16/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_PGD/1.attfuse_voxelNet_PGD_all_0.05_2023_09_02_16/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_xyz_0.07_2023_09_04_12/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_all_0.07_2023_09_04_12/attacked_pose'


#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_xyz_2023_09_05_08/attacked_pose'
# attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_all_2023_09_05_08/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_x_2023_09_07_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_y_2023_09_07_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_z_2023_09_07_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_roll_2023_09_07_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_yaw_2023_09_07_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_pitch_2023_09_07_08/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_0.07_x_2023_09_08_07/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_0.07_y_2023_09_08_07/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_0.07_z_2023_09_08_07/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_0.07_roll_2023_09_08_07/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_0.07_yaw_2023_09_08_07/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_0.07_pitch_2023_09_08_07/attacked_pose'


#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_multi_mmd/1.attfuse_voxelNet_multi_mmd_all_2023_08_29_17/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_PGD_all_2023_08_31_19/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_AdvGPS_shift_all_2023_09_09_16/attacked_pose'

#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_AdvGPS_mean+std_all_2023_09_09_16/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_AdvGPS_mean+2std_all_2023_09_09_16/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_AdvGPS_mean+3std_all_2023_09_09_16/attacked_pose'


#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_det_multi_mmd_shitf_based_PGD/1.attfuse_voxelNet_det_multi_mmd_shift_based_PGD_all_2023_09_05_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_random/1.attfuse_random_all_2023_08_22_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_FSGM/1.attfuse_voxelNet_FSGM_all_2023_08_31_08/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_IFSGM_0.05/1.attfuse_voxelNet_IFSGM_all_0.05_2023_09_02_16/attacked_pose'
#attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_PGD/1.attfuse_voxelNet_PGD_all_0.05_2023_09_02_16/attacked_pose'
attacked_pose_path='/home/jinlongli/2.model_saved/3.Attack_V2V2023/1.attfuse_voxelNet_SOK/1.attfuse_voxelNet_SOK_all_0.07_2023_09_04_12/attacked_pose'



###Attack
# model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/CoBEVT_finetun_voxel_2023_07_21_07/net_epoch7.pth"
######## using for oringinal point pillar training
CUDA_VISIBLE_DEVICES=1 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/test_attack_pose.py  --hypes_yaml $hypes_yaml --model $model --attacked_pose_path $attacked_pose_path
#--model_dir $path  #--model 

# CUDA_VISIBLE_DEVICES=6 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train.py  --hypes_yaml $hypes_yaml #--model $model #--model_dir $path  #--model 



# cd ../




# exit the virtual environment
# conda deactivate