# Optimizing a Wing

[Blog post](https://greydanus.github.io/2020/10/14/optimizing-a-wing/) | [Paper](https://bit.ly/3j3Wcu4) | [Colab notebook](https://bit.ly/2H5r401)

In this project, I use Navier-Stokes to simulate a wind tunnel, place a rectangular occlusion in it, and use gradient descent to optimize its lift/drag ratio. This gives us a wing shape. I'm releasing this repo as a supplement to a series of blog posts I wrote about human flight.

To obtain the figure below: clone this repo, `cd` into it, and run `python main.py `

![optimize_wing.png](./static/optimize_wing.png)

Note: the code and ideas in this repo build on [this Autograd demo](https://github.com/HIPS/autograd/blob/master/examples/fluidsim/wing.png).


## Appendix: Failure cases

**Biplane wing.** When making the differentiable region wider (in hopes of a wider wing) the wing split into two wings, like a biplane.

![biplane.gif](./static/biplane.gif)

**Shattered wing.** When adjusting the granularity of the simulation, we accidentally made the wing shatter into many little wings. Fun to look at, but probably not so fun to fly with.

![shatter.gif](./static/shatter.gif)

**Stubby wing.** This sad little wing occurred after making the initial rectangle too impermeable. Not enough simulated air was making it past the boundary layer of the rectangle so we hit a bad local minima.

![stubby.gif](./static/stubby.gif)
