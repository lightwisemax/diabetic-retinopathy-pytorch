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
from u_d.gan import gan
from u_d.update_d import update_d
from u_d.update_u import update_u