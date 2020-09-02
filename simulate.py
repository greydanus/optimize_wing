# Optimizing a Wing
# Sam Greydanus

import autograd
import autograd.numpy as np
import numpy as npo

import scipy
import scipy.ndimage


# Extend the Autograd library to include a differentiable Gaussian blur (see bit.ly/2YzMN6H)

@autograd.extend.primitive
def gaussian_filter(x, width):
  """Apply gaussian blur of a given radius."""
  return scipy.ndimage.gaussian_filter(x, width, mode='reflect')

def _gaussian_filter_vjp(ans, x, width):
  del ans, x  # unused
  return lambda g: gaussian_filter(g, width)
autograd.extend.defvjp(gaussian_filter, _gaussian_filter_vjp)



# A set of functions for simulating the wind tunnel

def occlude(x, occlusion):
  return x * (1 - occlusion)

def sigmoid(x):
  return 0.5*(np.tanh(x) + 1.0)

def advect(f, vx, vy):
  """Instead of moving the cell centers forward in time using the velocity fields,
  we look for the particles which end up exactly at the cell centers by tracing
  backwards in time from the cell centers. See 'implicit Euler integration.'"""
  rows, cols = f.shape
  cell_xs, cell_ys = np.meshgrid(np.arange(cols), np.arange(rows))
  center_xs = (cell_xs - vx).ravel()  # look backwards one timestep
  center_ys = (cell_ys - vy).ravel()

  left_ix = np.floor(center_ys).astype(int)  # get locations of source cells.
  top_ix  = np.floor(center_xs).astype(int)
  rw = center_ys - left_ix              # relative weight of cells on the right
  bw = center_xs - top_ix               # same for cells on the bottom
  left_ix  = np.mod(left_ix,     rows)  # wrap around edges of simulation.
  right_ix = np.mod(left_ix + 1, rows)
  top_ix   = np.mod(top_ix,      cols)
  bot_ix   = np.mod(top_ix  + 1, cols)

  # a linearly-weighted sum of the 4 cells closest to the source of the cell center.
  flat_f = (1 - rw) * ((1 - bw)*f[left_ix,  top_ix] + bw*f[left_ix,  bot_ix]) \
                + rw * ((1 - bw)*f[right_ix, top_ix] + bw*f[right_ix, bot_ix])
  return np.reshape(flat_f, (rows, cols))
  
def project(vx, vy, occlusion, width=0.4):
  """Project the velocity field to be approximately mass-conserving. Technically
  we are finding an approximate solution to the Poisson equation."""
  p = np.zeros(vx.shape)
  div = -0.5 * (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)
              + np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0))
  div = filter(div, occlusion, width=width)

  for k in range(50):  # use gauss-seidel to approximately solve linear system
      p = (div + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1)
                + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))/4.0
      p = filter(p, occlusion, width=width)

  vx = vx - 0.5*(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))
  vy = vy - 0.5*(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))

  vx = occlude(vx, occlusion)
  vy = occlude(vy, occlusion)
  return vx, vy

def filter(f, occlusion, width=0.4):
  '''Break f into parts inside the occlusion and outside the occlusion. Filter the
  latter part so that it diffuses into the occlusion and gives continuous boundary
  conditions. This makes the occlusion semipermeable and and easier to optimize.'''

  non_occluded = 1 - occlusion
  diffused_f = gaussian_filter(f*non_occluded, width) / gaussian_filter(non_occluded, width)

  outside_occlusion = non_occluded * f
  inside_occlusion = occlusion * diffused_f
  return outside_occlusion + inside_occlusion

def enforce_boundary_conditions(fields, names, args, k=3):
  '''Add boundary conditions that don't wrap around. This is a bit awkward because
      Autograd doesn't let us perform in-place assignment.'''
  rows, cols = fields[0].shape
  for i, (name, f) in enumerate(zip(names, fields)):
    top_wall, left_wall = np.zeros((k,cols)), np.zeros((rows,k))
    if name == 'red_smoke':
        left_wall[rows//4:rows//2] += 0.9  # red smoke comes out of left wall
    if name == 'blue_smoke':
        left_wall[rows//2:3*rows//4] += 0.9  # blue smoke comes out of left wall
    if name == 'vx':
        left_wall += args.wind_speed    # wind comes out of left wall
    f = np.concatenate([top_wall, f[k:]], axis=0)
    f = np.concatenate([left_wall, f[:,k:]], axis=1)
    fields[i] = f
  return fields

def constrain_occlusion(params, tunnel_shape, use_oval_shape=False):
  # constrain the occlusion to a particular shape and range of values
  rows, cols = tunnel_shape
  params = params.reshape(rows, cols)
  init_shape = np.zeros_like(params)
  if use_oval_shape:
    import numpy as npo
    er, ec, rad = int(rows*.5), int(cols*.45), int(rows*.45)
    y, x = npo.ogrid[-er:rows-er, -ec:cols-ec]
    init_shape[x**2 + 5*y**2 <= rad**2] = 1
  else:
    init_shape[int(.3*rows):int(.68*rows), int(.12*cols):int(.62*cols)] = 1.0
  
  return sigmoid(params) * init_shape

def simulate_wind_tunnel(args, vx, vy, occlusion):
  '''Code modified from bit.ly/2Yy8LXs. Physics is based on bit.ly/386n3BR'''
  rows, cols = args.tunnel_shape
  red_smoke, blue_smoke = np.zeros_like(vx), np.zeros_like(vx)  # add smoke
  red_smoke[rows//4:rows//2] = 0.9     # initialize red smoke band
  blue_smoke[rows//2:3*rows//4] = 0.9  # ...and blue smoke band
  rgb = np.stack([red_smoke, occlusion, blue_smoke], axis=-1)
  frames = [rgb]   # visualize occlusion and flow of smoke

  # Step through the simulation
  vx, vy = project(vx, vy, occlusion, args.filter_width)
  for t in range(args.simulator_steps):
    vx_updated = advect(vx, vx, vy)   # self-advection of vx
    vy_updated = advect(vy, vx, vy)   # self-advection of vy
    vx, vy = project(vx_updated, vy_updated, occlusion, args.filter_width)  # vol. constraint
    
    red_smoke = advect(red_smoke, vx, vy)   # advect / occlude the smoke
    red_smoke = occlude(red_smoke, occlusion)
    blue_smoke = advect(blue_smoke, vx, vy)
    blue_smoke = occlude(blue_smoke, occlusion)
    
    fields = ([red_smoke, blue_smoke, vx, vy], ['red_smoke', 'blue_smoke', 'vx', 'vy'])
    [red_smoke, blue_smoke, vx, vy] = enforce_boundary_conditions(*fields, args)

    rgb = np.stack([red_smoke, occlusion, blue_smoke], axis=-1)
    frames.append(rgb)
  return vx, vy, frames