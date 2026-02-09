cd /media/skr/storage/self_driving/CoPilot4D && nohup /media/skr/storage/conda_envs/nuscenes/bin/python -u scripts/train_tokenizer.py --config co
  nfigs/tokenizer_debug.yaml --resume outputs/tokenizer_debug/checkpoint_step_6000.pt --device cuda > training.log 2>&1 &

  