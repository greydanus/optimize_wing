# Optimizing a Wing
# Sam Greydanus

import matplotlib.pyplot as plt
from optimize import get_args, optimize_wing

if __name__ == '__main__':

  #### CONFIGURE AND RUN OPTIMIZATION ####
  args = get_args()
  simulations, final_params = optimize_wing(args)

  #### VISUALIZE THE RESULTS ####
  fig = plt.figure(figsize=(10, 3), dpi=100)

  plt.subplot(1,3,1) ; plt.title('Initial setup')
  plt.imshow(simulations[0][-1]) ; plt.xticks([],[]) ; plt.yticks([],[])

  plt.subplot(1,3,2) ; plt.title('During optimization')
  plt.imshow(simulations[5][-1]) ; plt.xticks([],[]) ; plt.yticks([],[])

  plt.subplot(1,3,3) ; plt.title('Final result')
  plt.imshow(simulations[-1][-1]) ; plt.xticks([],[]) ; plt.yticks([],[])

  plt.tight_layout() ; plt.show()
  fig.savefig('./optimize_wing.png')