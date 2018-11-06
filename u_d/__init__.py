"""
different model architecture for u+d
directory tree:
--training
  --analysis
    --dcgan
  --wgan_gp
note we mustn't put models in .train() or eval() models during training gan.
reference: https://discuss.pytorch.org/t/why-dont-we-put-models-in-train-or-eval-modes-in-dcgan-example/7422
"""
from u_d.wgan_gp import wgan_gp
