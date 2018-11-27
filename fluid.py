import numpy as np
import itertools
import matplotlib.pyplot as plt


def generate_position_indices(n_cells):
    return np.indices(n_cells).transpose(np.roll(range(len(n_cells) + 1), -1))


class Fluid:
    def __init__(self, size=50):
        self.size = size
        self.shape = (size, size)

        # ----- Simulation constants -------
        self.viscosity = 0.01
        self.dissipation = 0.0

        self.dt = 0.01
        # ----------------------------------

        self.velocity = np.zeros((size, size, 2))
        self.forces = np.zeros((size, size, 2))

        self.density = np.zeros(self.shape+(3,))
        self.sources = np.zeros(self.shape+(3,))

        self.position_indices = generate_position_indices(self.shape)

    def plot(self):
        plt.imshow(self.density[:,:,0])
        plt.colorbar()
        plt.quiver(self.velocity[:, :, 0], -self.velocity[:, :, 1], )
        plt.show()

    @staticmethod
    def set_boundaries(to_set):
        """
        For a vector field, we want the boundary to exercise a reaction force (opposite to the next cell)
        For strict edges, take the opposite value of next cell
        For corners take mean of the two adjacent
        """

        to_set[0, 1:-1, 0] = -to_set[1, 1:-1, 0]
        to_set[-1, 1:-1, 0] = -to_set[-2, 1:-1, 0]

        to_set[1:-1, 0, 1] = -to_set[1:-1, 1, 1]
        to_set[1:-1, -1, 1] = -to_set[1:-1, -2, 1]

        to_set[0, 0] = (to_set[0, 1] + to_set[1, 0]) / 2
        to_set[-1, 0] = (to_set[-1, 1] + to_set[-2, 0]) / 2
        to_set[0, -1] = (to_set[0, -2] + to_set[1, -1]) / 2
        to_set[-1, -1] = (to_set[-1, -2] + to_set[-2, -1]) / 2

    """
    #############################################################################
    Core methods: 
        - Advection
        - Diffusion
        - 
    """

    def advect(self, to_advect):
        """
        Advects an array given velocity, both array shapes must match the PhysicEngine n_cell config
        """

        positions = self.position_indices
        old_positions = positions - self.dt * self.size * self.velocity

        # Clip coordinates
        old_positions[:, :, 0] = np.clip(old_positions[:, :, 0], 0.5, self.shape[0] - 1.5)
        old_positions[:, :, 1] = np.clip(old_positions[:, :, 1], 0.5, self.shape[1] - 1.5)

        # Interpolation
        old_positions_int = old_positions.astype(int)

        alpha = 1 - (old_positions - old_positions_int)

        # Reshape alpha from (n,n,2) --> (n,n,2,2)
        if len(to_advect.shape) == 3:
            alpha = np.transpose(np.array([alpha]*to_advect.shape[2]), [1, 2, 3, 0])

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
        # We know the value in the middle of each cell, 2d interpolation for X

        advected = np.zeros(to_advect.shape)
        f = lambda x, y: 1 - x if y else x
        for i, j in itertools.product([0, 1], repeat=2):
            contribution = to_advect[old_positions_int[:, :, 0] + i, old_positions_int[:, :, 1] + j]
            advected += contribution * f(alpha[:, :, 0], i) * f(alpha[:, :, 1], j)

        return advected

    def diffuse(self, to_diffuse, n_steps: int = 20):
        alpha = self.dt * self.viscosity * np.prod(self.shape)

        diffused = np.zeros_like(to_diffuse)

        for _ in range(n_steps):
            diffused[1:-1, 1:-1] = (to_diffuse[1:-1, 1:-1] + alpha * (
                    diffused[:-2, 1:-1] +
                    diffused[1:-1, 2:] +
                    diffused[2:, 1:-1] +
                    diffused[1:-1, :-2])) / (1 + 4 * alpha)
            # TODO check boundaries

        return diffused

    def project(self, to_project, n_steps: int = 20):
        """
        Project the velocity field onto a mass conserving field
        (ie) with div F = 0
        (ie) incompressible ie in a cell, entering matter = leaving matter
        """
        divergence = np.zeros(to_project.shape[:-1])
        divergence[1:-1, 1:-1] = 0.5 * (
                to_project[:-2, 1:-1, 0] - to_project[2:, 1:-1, 0] +
                to_project[1:-1, :-2, 1] - to_project[1:-1, 2:, 1]) / self.size

        # Boundaries are already set to 0 .. TODO

        diffused_div = np.zeros_like(divergence)
        for _ in range(n_steps):
            diffused_div[1:-1, 1:-1] = (divergence[1:-1, 1:-1] + (
                    diffused_div[:-2, 1:-1] +
                    diffused_div[1:-1, 2:] +
                    diffused_div[2:, 1:-1] +
                    diffused_div[1:-1, :-2])) / 4

        # Update velocities
        projected = np.zeros_like(to_project)
        projected[1:-1, 1:-1, 0] = to_project[1:-1, 1:-1, 0] - 0.5 * self.size * (
                diffused_div[2:, 1:-1] - diffused_div[:-2, 1:-1])
        projected[1:-1, 1:-1, 1] = to_project[1:-1, 1:-1, 1] - 0.5 * self.size * (
                diffused_div[1:-1, 2:] - diffused_div[1:-1, :-2])

        self.set_boundaries(projected)
        return projected

    def dissipate(self, density):
        return density / (1 + self.dt * self.dissipation)

    def run(self):
        self.velocity = self.add_sources(self.velocity, self.forces)
        self.velocity = self.advect(self.velocity)
        self.velocity = self.diffuse(self.velocity)
        self.velocity = self.project(self.velocity)

        self.density = self.add_sources(self.density, self.sources)
        self.density = self.advect(self.density)
        #self.density = self.diffuse(self.density)
        self.density = self.dissipate(self.density)

    @staticmethod
    def add_sources(field, sources):
        return field + sources
