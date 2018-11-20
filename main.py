from fluid import Fluid
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fluid = Fluid()

    fluid.density[10, 10] = 1
    fluid.plot()

    fluid.velocity.fill(5*0.4142)

    for _ in range(100):
        fluid.run()
        fluid.velocity[20,20] = [-2,2]
        fluid.plot()

