# Spatial size of training images. All images will be resized to this
#   size using a transformer.
img_size = 64

# Number of channels in the training images. For color images this is 3
channels = 3

# Size of z latent vector (i.e. size of generator input)
latent_dim = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
n_epochs = 5

batch_size = 256

# Learning rate for optimizers
lr = 0.0002

# Beta hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.999

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

n_cpu = 2

n_critic = 5

clip_value = 0.01

sample_interval = 400

# Hyperparameters
hyperparams = {'lr' : [0.0002, 0.002, 0.02, 0.1],
               'n_critic' : [5, 10, 15]}

#{'D' : [0.0002, 0.002, 0.02, 0.1],
#                       'G' : [0.0002, 0.002, 0.02, 0.1]},

