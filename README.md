# Optimizing a Wing

[Paper](https://greydanus.github.io/2020/10/14/optimizing-a-wing/) | [Blog post](https://greydanus.github.io/2020/10/14/optimizing-a-wing/) | [Colab notebook](https://colab.research.google.com/drive/1RTsSyr7B3THKVGp_44Oyh7rxBriOHzJ7)

In this project, I use Navier-Stokes to simulate a wind tunnel, place a rectangular occlusion in it, and use gradient descent to optimize its lift/drag ratio. This gives us a wing shape. I'm releasing this repo as a supplement to a series of blog posts I wrote about human flight.

To obtain the figure below: clone this repo, `cd` into it, and run `python main.py `

![optimize_wing.png](optimize_wing.png)

Note: the code and ideas in this repo build on [this Autograd demo](https://github.com/HIPS/autograd/blob/master/examples/fluidsim/wing.png).

### Appendix: Fun failure cases

* When making the wing region longer in the hopes of getting a wider wing, I got a surprising result: [two wings](https://drive.google.com/file/d/1rwnlMd6etLoWvdqvyOeOcQY5cERk7YmS/view?usp=sharing).
* I got [an even wackier result](https://drive.google.com/file/d/1aq-Cxvg4xwH7MOD4VJ57bUNimwx41L_2/view?usp=sharing) while playing around with the granularity of the simulation.
* An example of [bad local minima](https://drive.google.com/file/d/1hBEOMML5QKRj-M0dE1Fiz0Y829loHzo8/view?usp=sharing) due to making the initial wing region too impermeable to the air.