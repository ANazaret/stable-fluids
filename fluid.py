import numpy as np
import itertools
import matplotlib.pyplot as plt


def generate_position_indices(n_cells):
    return np.indices(n_cells).transpose(np.roll(range(len(n_cells)+1), -1))


class Fluid2d:
    def __init__(self, n_cells=(50, 50)):
        self.n_cells = n_cells

        # ----- Simulation constants -------
        self.viscosity = 10
        self.diffusion = 10
        self.dissipation = 10
        # ----------------------------------

        self.velocity = np.zeros(n_cells + (2,))
        self.density = np.zeros(n_cells)

        # self.density[2,2,2] = 1

    def plot(self):
        plt.imshow(self.density)


class PhysicsEngine:
    def __init__(self, fluid):
        self.fluid = fluid
        self.n_cells = fluid.n_cells
        self.positions = generate_positions(fluid.n_cells)

    def apply_boundaries_vector_field(self, matrix):
        """
        For a vector field, we want the boundary to exercize a reaction force (opposite to the next cell)
        For strict edges, take the opposite value of next cell
        For corners take mean of the two adjacent
        """

        matrix[0, 1:-1] = -matrix[1, 1:-1]
        matrix[-1, 1:-1] = -matrix[-2, 1:-1]

        matrix[1:-1, 0] = -matrix[1:-1, 1]
        matrix[1:-1, -1] = -matrix[1:-1, -2]

        matrix[0, 0] = (matrix[0, 1] + matrix[1, 0]) / 2
        matrix[-1, 0] = (matrix[-1, 1] + matrix[-2, 0]) / 2
        matrix[0, -1] = (matrix[0, -2] + matrix[1, -1]) / 2
        matrix[-1, -1] = (matrix[-1, -2] + matrix[-2, -1]) / 2

    def advects(self, advected, velocity, dt: float):
        """
        Advects a array given velocity, both array shapes must match the PhysicEngine n_cell config
        """

        positions = self.positions
        old_positions = positions - dt * velocity

        # Does dt need to be timed by n_cells ?

        for i in range(2):
            # Try with 0 and -1 to see if there is a problem
            old_positions[:, :, i] = np.clip(old_positions[:, :, i], 0.5, self.n_cells[i] - 1.5)

        # Interpolation
        old_positions_int0 = old_positions.astype(int)
        old_positions_int1 = old_positions_int0 + 1

        alpha = 1 - (old_positions - old_positions_int0)
        # Reshape alpha (if needed) to allow fast numpy multiplications
        # if len(advected.shape) > len(alpha.shape)-1:
        #    alpha = alpha.reshape((1,)+alpha.shape)
        old_positions = [old_positions_int0, old_positions_int1]

        # We need to interpolate between 4 cells: example in 2D with 2^2 = 4 cells
        #   |------|-------|
        #   |      |       |
        #   |  O   |   O   |
        #   |      |    X  |
        #   |------|-------|
        #   |      |       |
        #   |  O   |   O   |
        #   |      |       |
        #   |------|-------|
        # We know the value in the middle of each cell, 2d interploation for X

        new_values = np.zeros(advected.shape)

        f = lambda x, y: 1 - x if y else x

        for i, j in itertools.product([0, 1], repeat=2):
            contribution = advected[old_positions[i][:, :, 0], old_positions[j][:, :, 1]]
            new_values += contribution * f(alpha[:, :, 0], i) * f(alpha[:, :, 1], j)

        return new_values

    def projects(self, x):
        """
        Project the velocity fieled onto a mass conserving field (ie with div F = 0 ie incompressible ie in a cell, entering matter = leaving matter )
        """

    def run(self, dt):
        self.fluid.density = self.advects(self.fluid.density, self.fluid.velocity, dt)



