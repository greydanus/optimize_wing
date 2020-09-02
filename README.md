# Optimizing a Wing
Sam Greydanus

[Blog post](https://greydanus.github.io/2020/07/30/physics-of-flight/)

[Colab notebook](https://colab.research.google.com/drive/1RTsSyr7B3THKVGp_44Oyh7rxBriOHzJ7)

In this project, I use Navier-Stokes to simulate a wind tunnel, place a rectangular occlusion in it, and use gradient descent to optimize its lift/drag ratio. This gives us a wing shape. I'm releasing this repo as a supplement to a series of blog posts I wrote about human flight. Also note that this code was originally modified from [this Autograd demo](bit.ly/2Yy8LXs).

![setup.png](static/setup.png) ![wing.png](static/wing.png)