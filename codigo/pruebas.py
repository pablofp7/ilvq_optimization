import torch
import tensorflow as tf

x = torch.rand(2, 2)
y = torch.rand(2, 2)
x_tf = tf.random.uniform([2, 2])
y_tf = tf.random.uniform([2, 2])

print(f"Con Pytorch: {x}, con Tensorflow: {x_tf}")
print(f"Con Pytorch: {y}, con Tensorflow: {y_tf}")

z = x + y
z2 = torch.add(x, y)
z_tf = x_tf + y_tf
z2_tf = tf.add(x_tf, y_tf)

print(f"Con Pytorch: {z}, con Tensorflow: {z_tf}")
print(f"Con Pytorch: {z2}, con Tensorflow: {z2_tf}")


