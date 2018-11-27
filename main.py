from fluid import Fluid
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
from event_handler import EventHandler

from render import render_fluid

window_size = 600
fluid_size = 100
CELL_IN_PIXEL = window_size // fluid_size

if __name__ == '__main__':
    fluid = Fluid(fluid_size)

    fluid.sources[10:12, 10:12] = 1
    fluid.forces[10, 10] = [2, 2]
    fluid.plot()

    fluid.velocity[:, :, 0].fill(5 * 0.4142)
    fluid.velocity[:, :, 1].fill(5 * 0.4142)

    fluid.velocity[20, 20] = [-2, 2]

    pygame.init()
    window = pygame.display.set_mode([window_size, window_size])
    red = (230, 50, 50)

    event_handler = EventHandler(fluid, window, CELL_IN_PIXEL)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

            event_handler.handle_event(event)

        window.fill((40, 50, 180))
        fluid.run()
        render_fluid(fluid, window)

        pygame.display.update()
        print(fluid.density.sum())
