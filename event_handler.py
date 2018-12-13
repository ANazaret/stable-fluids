import pygame
from pygame.locals import *

import matplotlib.pyplot as plt
import numpy as np


class EventHandler:

    def __init__(self, fluid, window, cell_in_pixel):
        self.window = window
        self.fluid = fluid

        self.cell_in_pixel = cell_in_pixel
        self.left_button_is_down = False
        self.right_button_is_down = False

        self.color = 0

    def handle_event(self, event: pygame.event.EventType):
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                self.left_button_is_down = True
                self.fluid.sources[event.pos[0] // self.cell_in_pixel,
                                   event.pos[1] // self.cell_in_pixel,
                                   self.color] += 100
            elif event.button == 3:
                print("Right is down")
                self.right_button_is_down = True

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                self.left_button_is_down = False
            elif event.button == 3:
                self.right_button_is_down = False

        elif event.type == MOUSEMOTION:
            if self.right_button_is_down:
                self.fluid.forces[event.pos[0] // self.cell_in_pixel,
                                  event.pos[1] // self.cell_in_pixel] += np.array(event.rel) * 10

        elif event.type == KEYDOWN:
            if event.key == K_v:
                plt.quiver(self.fluid.velocity[:, :, 0], -self.fluid.velocity[:, :, 1], units='xy', scale=5)
                plt.show()
            elif event.key == K_f:
                plt.quiver(self.fluid.forces[:, :, 0], -self.fluid.forces[:, :, 1], units='xy', scale=5)
                plt.show()

            elif event.key == K_c:
                self.color = (self.color + 1) % 3
            elif event.key == K_UP:
                self.fluid.viscosity *= 1.5
            elif event.key == K_DOWN:
                self.fluid.viscosity /= 1.5
            elif event.key == K_r:
                self.fluid.forces *= 0
            elif event.key == K_s:
                self.fluid.velocity_boundaries = (
                    self.fluid.continuity_boundaries if self.fluid.velocity_boundaries == self.fluid.mirror_boundaries
                    else self.fluid.mirror_boundaries)
