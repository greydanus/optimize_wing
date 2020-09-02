# Optimizing a Wing
# Sam Greydanus

import autograd
import autograd.numpy as np
import numpy as npo

import time
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt

from .simulate import simulate_wind_tunnel, constrain_occlusion


class ObjectView(object):  # make a dictionary look like an object
  def __init__(self, d): self.__dict__ = d

def get_args(as_dict=False):
  arg_dict = {'tunnel_shape': [50, 75],
              'learning_rate': 1e3,
              'wind_speed': 1,
              'mass_coeff': 0,
              'noise_coeff': 1e-1,
              'print_every': 2,
              'filter_width': 1,
              'seed': 0,
              'use_oval_shape': False,
              'simulator_steps': 20,
              'optimization_steps': 20}
  return arg_dict if as_dict else ObjectView(arg_dict)


def optimize_wing(args):
  np.random.seed(args.seed)
  
  init_vx = args.wind_speed * np.ones(args.tunnel_shape)
  init_vy = np.zeros_like(init_vx)
  init_params = args.noise_coeff * np.random.rand(*init_vx.shape) - 1

  def get_lift_drag_ratio(occlusion):
    final_vx, final_vy, _ = simulate_wind_tunnel(args, init_vx, init_vy, occlusion)
    lift = -np.mean(final_vy - init_vy)
    drag = np.mean(final_vx - init_vx)
    return lift / drag

  def objective(params):
    occlusion = constrain_occlusion(params, args.tunnel_shape, args.use_oval_shape)
    ld_ratio = get_lift_drag_ratio(occlusion)
    mass_multiplier = args.mass_coeff * occlusion[occlusion>0].mean()
    return ld_ratio + mass_multiplier

  grad_fn = autograd.value_and_grad(objective)  # autograd magic
  params = init_params.ravel()

  # need to run simulation on initial conditions
  occlusion = constrain_occlusion(params, args.tunnel_shape, args.use_oval_shape)
  _, _, frames = simulate_wind_tunnel(args, init_vx, init_vy, occlusion)
  simulations = [frames]
  t0 = time.time()
  for step in range(args.optimization_steps):  # main optimization loop
    loss, grad = grad_fn(params)
    params += args.learning_rate * grad    # maximize lift/drag
    
    # logging
    occlusion = constrain_occlusion(params, args.tunnel_shape, args.use_oval_shape)
    _, _, frames = simulate_wind_tunnel(args, init_vx, init_vy, occlusion)
    simulations.append(np.stack(frames))
    if (step+1) % args.print_every == 0:
      print('step: {}, lift/drag ratio: {:.2e}, wallclock dt: {:.2f}s'.format(step+1, loss, time.time()-t0))
      t0 = time.time()
  return simulations, params