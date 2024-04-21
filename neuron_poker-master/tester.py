# from gymnasium import envs
# for key, value in envs.registry.items():
#     print(key)


import tensorflow as tf


print(tf.test.is_gpu_available())
print(tf.device('/device:GPU:0'))