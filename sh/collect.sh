python -m scr.collect PongDeterministic-v4 --min-burnin 50 --max-burnin 300 --max-episodes 10000 --num-steps 10 --save-path data/pong_full_train.h5 --seed 1
#python -m scr.collect PongDeterministic-v4 --min-burnin 50 --max-burnin 300 --max-episodes 10000 --num-steps 10 --save-path data/pong_full_eval.h5 --seed 2

python -m scr.collect SpaceInvadersDeterministic-v4 --min-burnin 50 --max-burnin 300 --max-episodes 10000 --num-steps 10 --save-path data/spaceinvaders_full_train.h5 --seed 1
#python -m scr.collect SpaceInvadersDeterministic-v4 --min-burnin 50 --max-burnin 300 --max-episodes 10000 --num-steps 10 --save-path data/spaceinvaders_full_eval.h5 --seed 2
