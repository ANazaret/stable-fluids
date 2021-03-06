import numpy as np

import pygame
import matplotlib.image as mpimg
from pygame.locals import *

from event_handler import EventHandler
from fluid import Fluid
from render import render_fluid

window_size = 1000
fluid_size = 100
CELL_IN_PIXEL = window_size // fluid_size

if __name__ == '__main__':
    fluid = Fluid(fluid_size)

    # fluid.sources[38:42, 39:42, 0] = 100
    # fluid.forces[40:42, 40:42] = [-400, 150]

    # fluid.forces[10, 10] = [2, 2]
    # fluid.sources[80:82, 80:82, 1] = 100
    # fluid.forces[10, 10] = [2, 2]

    # fluid.velocity[:, :, 0].fill(5 * 0.4142)
    # fluid.velocity[:, :, 1].fill(5 * 0.4142)

    # fluid.velocity[20, 20] = [-2, 2]

    image = mpimg.imread('apr.png')
    # fluid.density = image[:,:,:3].transpose([1,0,2])
    fluid.viscosity = 0.0001
    fluid.density[0, :] = 0
    fluid.density[-1, :] = 0
    fluid.density[:, 0] = 0
    fluid.density[:, -1] = 0

    #fluid.forces += [8,20]

    #fluid.particles = np.random.randint(5, 40, (500, 2)).astype(float)
    # np.array([[10., 12], [42, 42]])

    pygame.init()
    window = pygame.display.set_mode([window_size, window_size])
    red = (230, 50, 50)

    event_handler = EventHandler(fluid, window, CELL_IN_PIXEL)

    quantity = []

    loop = True
    while loop:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                loop = False
            event_handler.handle_event(event)
        if not loop:
            break

        window.fill((40, 50, 180))
        fluid.run()
        render_fluid(fluid, window)

        pygame.display.update()
        # pygame.time.wait(30)
        quantity.append(fluid.density.sum())

    import matplotlib.pyplot as plt

    plt.plot(quantity)
    plt.xlabel("time (0.04s)")
    plt.ylabel("quantity of matter")
    plt.show()
