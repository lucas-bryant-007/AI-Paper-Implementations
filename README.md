# AI-Paper-Implementations
Implementations of AI Papers from scratch with the purpose of learning and understanding these algorithms and architectures.


## TODO
- Implement better action space and obs space handling for PPO so that it can take nearly any size and shape input and output any shape action (Definition of Done is all Gym Environments accepted and actually trainable on)
- Fix terminated vs truncated. Time-limit truncations should bootstrap instead of zeroing continuation value (PPO)
- Mini-Batch sampling (PPO)
- Vectorized envs for faster training of algorithms