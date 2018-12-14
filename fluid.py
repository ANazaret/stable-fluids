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
        self.viscosity = 0.005
        self.dissipation = 0.0

        self.dt = 0.01
        # ----------------------------------

        self.velocity = np.zeros((size, size, 2))
        self.forces = np.zeros((size, size, 2))

        self.density = np.zeros(self.shape + (3,))
        self.sources = np.zeros(self.shape + (3,))

        self.position_indices = generate_position_indices(self.shape)

        self.velocity_boundaries = self.mirror_boundaries

        self.particles = np.array([])
        self.radius = 2.5
        self.marching_grid = 200
        self.marching_step = 5
        self.edges = []

    def plot(self):
        plt.imshow(self.density[:, :, 0])
        plt.colorbar()
        plt.quiver(self.velocity[:, :, 0], -self.velocity[:, :, 1], )
        plt.show()

    @staticmethod
    def mirror_boundaries(to_set):
        """
        For a vector field, we want the boundary to exercise a reaction force (opposite to the next cell)
        For strict edges, take the opposite value of next cell
        For corner, we could average, but it not really important
        """

        to_set[0, 1:-1, 0] = -to_set[1, 1:-1, 0]
        to_set[-1, 1:-1, 0] = -to_set[-2, 1:-1, 0]

        to_set[1:-1, 0, 1] = -to_set[1:-1, 1, 1]
        to_set[1:-1, -1, 1] = -to_set[1:-1, -2, 1]

    @staticmethod
    def zero_normal_boundaries(to_set):
        """
        For a vector field, we want the boundary to exercise a reaction force (opposite to the next cell)
        For strict edges, take the opposite value of next cell
        For corner, we could average, but it is not really important
        """

        Fluid.continuity_boundaries(to_set)
        to_set[0, 1:-1, 0] = 0
        to_set[-1, 1:-1, 0] = 0

        to_set[1:-1, 0, 1] = 0
        to_set[1:-1, -1, 1] = 0

    @staticmethod
    def continuity_boundaries(to_set):
        """

        """
        to_set[0, :] = to_set[1, :]
        to_set[-1, :] = to_set[-2, :]

        to_set[:, 0] = to_set[:, 1]
        to_set[:, -1] = to_set[:, -2]

    def interpolate(self, positions, original):
        # Interpolation
        positions_int = positions.astype(int)

        # Clip coordinates to lower left cell id
        positions_int = np.clip(positions_int, 0, self.size - 2)

        alpha = np.clip(1 - (positions - positions_int), 0, 1)

        # Reshape alpha from (n,n,k) --> (n,n,k,k) if necessary (numpy dimension error)
        if len(original.shape) == 3:
            alpha = np.transpose(np.array([alpha] * original.shape[2]), [1, 2, 3, 0])

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

        interpolated = np.zeros(original.shape)
        f = lambda x, y: 1 - x if y else x
        for i, j in itertools.product([0, 1], repeat=2):
            contribution = original[positions_int[:, :, 0] + i, positions_int[:, :, 1] + j]
            interpolated += contribution * f(alpha[:, :, 0], i) * f(alpha[:, :, 1], j)

        return interpolated

    def interpolate_list(self, positions, original):
        # Interpolation
        positions_int = positions.astype(int)

        # Clip coordinates to lower left cell id
        positions_int = np.clip(positions_int, 0, self.size - 2)
        alpha = np.clip(1 - (positions - positions_int), 0, 1)

        interpolated = np.zeros(alpha.shape)
        alpha = np.transpose(np.array([alpha] * original.shape[2]), [1, 2, 0])
        f = lambda x, y: 1 - x if y else x
        for i, j in itertools.product([0, 1], repeat=2):
            contribution = original[positions_int[:, 0] + i, positions_int[:, 1] + j]
            interpolated += contribution * f(alpha[:, 0], i) * f(alpha[:, 1], j)

        return interpolated

    def advect(self, to_advect, continuity=lambda x: None):
        """
        Advects an array given velocity
        """

        positions = self.position_indices

        # ##############  Classic backtrack ############
        # old_positions = positions - self.dt * self.size * self.velocity
        # advected = self.interpolate(old_positions, to_advect)
        # ##############################################

        # Runge Kutta 2nd order
        middle_velocity = self.interpolate(positions - self.dt * self.size * self.velocity / 2.,
                                           self.velocity)
        old_positions = positions - self.dt * self.size * middle_velocity
        advected = self.interpolate(old_positions, to_advect)

        continuity(advected)

        return advected

    def diffuse(self, to_diffuse, continuity=lambda x: None, n_steps: int = 10):
        alpha = self.dt * self.viscosity * np.prod(self.shape)
        # alpha = 0.85/4
        diffused = np.zeros_like(to_diffuse)

        for _ in range(n_steps):
            diffused[1:-1, 1:-1] = (to_diffuse[1:-1, 1:-1] + alpha * (
                    diffused[:-2, 1:-1] +
                    diffused[1:-1, 2:] +
                    diffused[2:, 1:-1] +
                    diffused[1:-1, :-2])) / (1 + 4 * alpha)

            continuity(diffused)

        return diffused

    def project(self, to_project, n_steps: int = 20):
        """
        Project the velocity field onto a mass conserving field
        (ie) with div F = 0
        (ie) incompressible ie in a cell, entering matter = leaving matter
        """
        divergence = np.zeros(to_project.shape[:-1])
        divergence[1:-1, 1:-1] = (
                                         to_project[:-2, 1:-1, 0] - to_project[2:, 1:-1, 0] +
                                         to_project[1:-1, :-2, 1] - to_project[1:-1, 2:, 1]) / (2. * self.size)
        self.continuity_boundaries(divergence)

        diffused_div = np.zeros_like(divergence)
        for _ in range(n_steps):
            diffused_div[1:-1, 1:-1] = (divergence[1:-1, 1:-1] +
                                        diffused_div[:-2, 1:-1] +
                                        diffused_div[1:-1, 2:] +
                                        diffused_div[2:, 1:-1] +
                                        diffused_div[1:-1, :-2]) / 4

            self.continuity_boundaries(diffused_div)

        # Update velocities
        projected = np.zeros_like(to_project)
        projected[1:-1, 1:-1, 0] = to_project[1:-1, 1:-1, 0] - 0.5 * self.size * (
                diffused_div[2:, 1:-1] - diffused_div[:-2, 1:-1])
        projected[1:-1, 1:-1, 1] = to_project[1:-1, 1:-1, 1] - 0.5 * self.size * (
                diffused_div[1:-1, 2:] - diffused_div[1:-1, :-2])

        self.mirror_boundaries(projected)
        return projected

    def dissipate(self, density):
        return density / (1 + self.dt * self.dissipation)

    def run(self):
        if len(self.particles):
            self.update_particles()

        self.velocity = self.add_sources(self.velocity, self.forces)
        self.velocity = self.advect(self.velocity, self.velocity_boundaries)
        self.velocity = self.diffuse(self.velocity, self.velocity_boundaries)
        self.velocity = self.project(self.velocity)

        self.density = self.add_sources(self.density, self.sources)
        self.density = self.advect(self.density, self.continuity_boundaries)
        # self.density = self.diffuse(self.density, self.continuity_boundaries)
        # self.density = self.dissipate(self.density)

    def add_sources(self, field, sources):
        return field + sources * self.dt

    def update_particles(self):
        speed = self.interpolate_list(self.particles, self.velocity)
        self.particles += self.dt * speed * 20
        self.marching_square()

    def get_lattice_in_circle(self, r, x, y):
        int_x = int(x)
        int_y = int(y)
        int_r = int(r) + 1

        res = []

        for xi in range(max(int_x - int_r, 0), int_x + int_r + 1):
            for yi in range(max(0, int_y - int_r), int_y + int_r + 1):
                d = (xi - x) ** 2 + (yi - y) ** 2 - r ** 2
                if d < 0:
                    res.append([xi, yi, d])

        return res

    def marching_square(self):
        grid = np.zeros((self.marching_grid, self.marching_grid))
        grid_bool = np.zeros((self.marching_grid, self.marching_grid), dtype=bool)

        ratio = self.marching_grid / self.size

        # Here we don't have many particules, faster to go through each of them and validate points
        for particule in self.particles:
            good = np.array(self.get_lattice_in_circle(self.radius, *(particule * ratio)))
            if len(good):
                grid[good[:, 0].astype(int), good[:, 1].astype(int)] += good[:, 2]
                grid_bool[good[:, 0].astype(int), good[:, 1].astype(int)] |= True

        edges = []

        grid_kind = np.zeros((self.marching_grid - 1, self.marching_grid - 1), dtype=int)
        grid_kind = 8 * grid_bool[:-1, :-1] + 4 * grid_bool[1:, :-1] + 2 * grid_bool[1:, 1:] + grid_bool[:-1, 1:]
        for i in range(self.marching_grid - 1):
            for j in range(self.marching_grid - 1):
                kind = grid_kind[i, j]
                if kind == 0b1110 or kind == 0b0001:
                    edges.append([(i, j + 0.5), (i + 0.5, j + 1)])
                elif kind == 0b1101 or kind == 0b0010:
                    edges.append([(i + 1, j + 0.5), (i + 0.5, j + 1)])
                elif kind == 0b1011 or kind == 0b0100:
                    edges.append([(i + 1, j + 0.5), (i + 0.5, j)])
                elif kind == 0b1101 or kind == 0b0010:
                    edges.append([(i, j + 0.5), (i + 0.5, j)])
                elif kind == 0b1100 or kind == 0b0011:
                    edges.append([(i, j + 0.5), (i + 1, j + 0.5)])
                elif kind == 0b0110 or kind == 0b1001:
                    edges.append([(i + 0.5, j), (i + 0.5, j + 1)])
                elif kind == 0b0101:
                    mean = grid[i:i + 2, j:j + 2].mean()
                    if mean < 0:
                        edges.append([(i, j + 0.5), (i + 0.5, j + 1)])
                        edges.append([(i + 1, j + 0.5), (i + 0.5, j)])
                    else:
                        edges.append([(i, j + 0.5), (i + 0.5, j )])
                        edges.append([(i + 1, j + 0.5), (i + 0.5, j+1)])
                elif kind == 0b1010:
                    mean = grid[i:i + 2, j:j + 2].mean()
                    if mean > 0:
                        edges.append([(i, j + 0.5), (i + 0.5, j + 1)])
                        edges.append([(i + 1, j + 0.5), (i + 0.5, j)])
                    else:
                        edges.append([(i, j + 0.5), (i + 0.5, j )])
                        edges.append([(i + 1, j + 0.5), (i + 0.5, j+1)])
        self.edges = edges
