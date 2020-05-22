#!/bin/bash
cd ..
python train.py --batch_size=64 --lr_actor=0.00001 --lr_critic=0.00010 --model_path=01Run 
python train.py --batch_size=128 --lr_actor=0.00001 --lr_critic=0.00010 --model_path=02Run 
python train.py --batch_size=64 --lr_actor=0.0001 --lr_critic=0.00010 --model_path=03Run 
python train.py --batch_size=128 --lr_actor=0.0001 --lr_critic=0.00010 --model_path=04Run 
python train.py --batch_size=64 --lr_actor=0.0001 --lr_critic=0.00010 --actor_layer_dim_3=64 --model_path=05Run 
python train.py --batch_size=128 --lr_actor=0.0001 --lr_critic=0.00010 --actor_layer_dim_3=64 --model_path=06Run
python train.py --batch_size=256 --lr_actor=0.0001 --lr_critic=0.00010 --actor_layer_dim_3=64 --model_path=07Run
python train.py --batch_size=64 --lr_actor=0.0001 --lr_critic=0.00010 --actor_layer_dim_3=128 --model_path=08Run
python train.py --batch_size=64 --lr_actor=0.00001 --lr_critic=0.00010 --actor_layer_dim_3=128 --model_path=09Run
python train.py --batch_size=64 --lr_actor=0.0001 --lr_critic=0.00001 --critic_layer_dim_3=64 --model_path=10Run
