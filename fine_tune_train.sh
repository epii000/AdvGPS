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
# hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_fax.yaml"
#hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_transformer.yaml"
#hypes_yaml="/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/point_pillar_fcooper.yaml"
hypes_yaml='/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/hypes_yaml/voxelnet_intermediate_fusion.yaml'

#model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/attfuse/net_epoch20.pth"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/V2VAM/net_epoch52.pth"
#model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT/net_epoch60.pth"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2x-vit/net_epoch25.pth"

#model_dir="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/attfuse_fintune_voxel_2023_07_21_07"
#model_dir="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/V2VAM_finetune_voxel_2023_07_21_07"
# model_dir="/home/jinlongli/2.model_saved/3.Attack_V2V2023/CoBEVT_re_train_2023_08_18_12"
#model_dir="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/v2xvit_finetune_voxel_2023_07_21_10"
#model_dir="/home/jinlongli/2.model_saved/3.Attack_V2V2023/0.attack_model/f-cooper"
# model_dir='/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/checkpoint/voxelnet_attentive_fusion'
model='/home/jinlongli/1.Detection_Set/V2V_Attack/opencood/checkpoint/voxelnet_attentive_fusion/latest.pth'


###Attack
# model="/home/jinlongli/2.model_saved/3.Attack_V2V2023/CoBEVT_finetun_voxel_2023_07_21_07/net_epoch7.pth"

######## using for oringinal point pillar training
CUDA_VISIBLE_DEVICES=2 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train.py  --hypes_yaml $hypes_yaml  --model $model #--model_dir $model_dir  #--model 

# CUDA_VISIBLE_DEVICES=6 python3 /home/jinlongli/1.Detection_Set/V2V_Attack/opencood/tools/train.py  --hypes_yaml $hypes_yaml #--model $model #--model_dir $path  #--model 



# cd ../




# exit the virtual environment
# conda deactivate