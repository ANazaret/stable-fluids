import pygame
from pygame.locals import *

import numpy as np
from fluid import Fluid


def get_color(density):
    return (min(int(density * 255), 255), 0, 0)


def render_fluid(fluid: Fluid, window: pygame.Surface):
    cell_in_pixels = window.get_height() // fluid.size
    for i in range(fluid.size):
        for j in range(fluid.size):
            color = get_color(fluid.density[i, j])
            pygame.draw.rect(window, color,
                             Rect((i * cell_in_pixels, j * cell_in_pixels), (cell_in_pixels, cell_in_pixels)))