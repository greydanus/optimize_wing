# Optimizing a Wing

[Paper](https://greydanus.github.io/2020/10/14/optimizing-a-wing/) | [Blog post](https://greydanus.github.io/2020/10/14/optimizing-a-wing/) | [Colab notebook](https://colab.research.google.com/drive/1RTsSyr7B3THKVGp_44Oyh7rxBriOHzJ7)

In this project, I use Navier-Stokes to simulate a wind tunnel, place a rectangular occlusion in it, and use gradient descent to optimize its lift/drag ratio. This gives us a wing shape. I'm releasing this repo as a supplement to a series of blog posts I wrote about human flight.

To obtain the figure below: clone this repo, `cd` into it, and run `python main.py `

![optimize_wing.png](optimize_wing.png)

Note: the code and ideas in this repo build on [this Autograd demo](https://github.com/HIPS/autograd/blob/master/examples/fluidsim/wing.png).

### Appendix: Fun failure cases

* [Biplane wing](https://drive.google.com/file/d/1rwnlMd6etLoWvdqvyOeOcQY5cERk7YmS/view?usp=sharing): when making the differentiable region wider (in hopes of a wider wing) the wing split into two wings, like a biplane.

* [Broken wing](https://drive.google.com/file/d/1aq-Cxvg4xwH7MOD4VJ57bUNimwx41L_2/view?usp=sharing): When adjusting the granularity of the simulation, we accidentally made the wing shatter into many little wings. Fun to look at, but probably not so fun to fly with.

* [Stubby wing](https://drive.google.com/file/d/1rwnlMd6etLoWvdqvyOeOcQY5cERk7YmS/view?usp=sharing): This sad little wing occurred after making the initial rectangle too impermeable (not enough flow was entering the middle of the rectangle, so we hit a bad local minima)
