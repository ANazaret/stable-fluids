import pygame
from pygame.locals import *

import numpy as np
from fluid import Fluid


def get_color(density):
    return (min(int(density[0] * 255), 255),
            min(int(density[1] * 255), 255),
            min(int(density[2] * 255), 255))


def render_fluid(fluid: Fluid, window: pygame.Surface):
    cell_in_pixels = window.get_height() // fluid.size

    for i in range(fluid.size):
        for j in range(fluid.size):
            # color = get_color(fluid.density[i, j])
            color = get_color(fluid.density[i, j])
            pygame.draw.rect(window, color,
                             Rect((i * cell_in_pixels, j * cell_in_pixels), (cell_in_pixels, cell_in_pixels)))
    for p in fluid.particles:
        pygame.draw.circle(window, (125,230,20), (p*cell_in_pixels).astype(int), 2)

    for (xs, ys),(xe, ye) in fluid.edges:
        pygame.draw.line(window, (255,0,0), (xs*5-2.5, ys*5-2.5), (xe*5-2.5, ye*5-2.5) )