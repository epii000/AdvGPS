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


hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/voxelnet_intermediate_fusion.yaml"
#hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_opv2v.yaml"
# hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml"
# hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_fax.yaml"
# hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_transformer.yaml"


#model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/attfuse/net_epoch20.pth"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/V2VAM/net_epoch52.pth"
#model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT/net_epoch60.pth"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2x-vit/net_epoch25.pth"

#model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/attfuse_fintune_voxel_2023_07_21_07/net_epoch31.pth"
model='/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/attfuse_voxelNet_finetune_2023_08_23_22/net_epoch30.pth'

###Attack
# model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/CoBEVT_finetun_voxel_2023_07_21_07/net_epoch7.pth"

######## using for oringinal point pillar training
# CUDA_VISIBLE_DEVICES=6 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train_attack_pose.py  --hypes_yaml $hypes_yaml --model $model #--model_dir $path  #--model 

# CUDA_VISIBLE_DEVICES=6 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train.py  --hypes_yaml $hypes_yaml #--model $model #--model_dir $path  #--model 



#### MMD Attack
#CUDA_VISIBLE_DEVICES=1 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train_attack_pose_IFSGM.py  --hypes_yaml $hypes_yaml --model $model #--model_dir $path  #--model
#CUDA_VISIBLE_DEVICES=5 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train_attack_pose_mmd.py  --hypes_yaml $hypes_yaml --model $model 
#CUDA_VISIBLE_DEVICES=5 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train_attack_pose_multi_mmd.py  --hypes_yaml $hypes_yaml --model $model
#CUDA_VISIBLE_DEVICES=3 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train_attack_pose_PGD.py  --hypes_yaml $hypes_yaml --model $model
#CUDA_VISIBLE_DEVICES=2 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train_attack_pose_FSGM.py  --hypes_yaml $hypes_yaml --model $model #--model_dir $path  #--model
CUDA_VISIBLE_DEVICES=7 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train_attack_pose_SOK.py  --hypes_yaml $hypes_yaml --model $model


# cd ../




# exit the virtual environment
# conda deactivate