from matplotlib import pyplot as plt
import numpy as np

class Lens:
    """
    Represents a lens in a 2D space.

    Attributes:
        x (np.array): The x coordinates of the lens.
        y (np.array): The y coordinates of the lens.
        dx (np.array): The derivative of the lens.
        refidx_l (float): The refractive index on the left side of the lens.
        refidx_r (float): The refractive index on the right side of the lens.
        refl_coeff (float): The reflection coefficient.
    """
    def __init__(
        self,
        x: np.array,
        y: np.array,
        dx: np.array,
        refidx_l: float = 1,
        refidx_r: float = 1,
        refl_coeff: float = 0.1,
        trans_coeff: float = 0.9,
    ):

        self.x = x
        self.y = y
        self.dx = dx

        self.refidx_l = refidx_l
        self.refidx_r = refidx_r

        self.refl_coeff = refl_coeff
        self.trans_coeff = trans_coeff

        self.points = []
        self.segments = []
        self.bounding_box = []

        self.update()

    def update(self):
        """
        Updates the lens.
        """
        self.calculate_segments()
        self.calculate_points()
        self.calculate_bounds()

    @classmethod
    def lens_asphere(
        cls,
        res: int,
        r: float,
        k: float,
        a: list,
        refidx_l: float = 1,
        refidx_r: float = 1,
        refl_coeff: float = 0.01,
        trans_coeff: float = 0.9,
    ):    
        """
        Creates an aspheric lens.

        Args:
            res (int): The resolution of the lens.
            r (float): The radius of the lens.
            k (float): The conic constant.
            a (list): The additional terms.
            refidx_l (float): The refractive index on the left side of the lens.
            refidx_r (float): The refractive index on the right side of the lens.
            refl_coeff (float): The reflection coefficient.

        Returns:
            Lens: The aspheric lens.
        """

        # Sample the -1 to 1 range with equidistant points
        y = np.linspace(-1, 1, res)

        sqrt_el = np.sqrt(1 - ((1 + k) * y**2) / r**2)

        # Aspheric lens formula
        x = y**2 / (r * (1 + sqrt_el)) - 5

        # Additional terms
        for i in range(len(a)):
            x += a[i] * y ** (4 + i * 2)

        # Derivative
        # https://www.wolframalpha.com/input?i=derivative+of+x%28y%29+%3D+y%5E2+%2F+%28R+%281+%2B+sqrt%281+-+%28%281+%2B+k%29+y%5E2%29+%2F+R%5E2%29%29%29+%2B+a4+y%5E4+%2B+a6+y%5E6
        dx = 2 * y / (r * (sqrt_el + 1)) + ((k + 1) * y**3) / (
            r**3 * (sqrt_el) * (sqrt_el + 1) ** 2
        )

        # Additional terms
        for i in range(len(a)):
            dx += (4 + i * 2) * a[i] * y ** (4 + i * 2 - 1)

        dx = -dx

        # Move the lense to the origin
        x -= min(x)

        return cls(x, y, dx, refidx_l, refidx_r, refl_coeff, trans_coeff)

    @classmethod
    def lens_segment(
        cls,
        res: int,
        start: np.array,
        end: np.array,
        refidx_l: float = 1,
        refidx_r: float = 1,
        refl_coeff: float = 0.01,
        trans_coeff: float = 0.9,
    ):

        x = np.linspace(start[0], end[0], res)
        y = np.linspace(start[1], end[1], res)
        if end[1] == start[1]:
            if start[0] > end[0]:
                dx = np.array(res * np.inf)
            else:
                dx = -np.array(res * np.inf)
        else:
            dx = np.array(res * [(start[0] - end[0]) / (end[1] - start[1])])

        return cls(x, y, dx, refidx_l, refidx_r, refl_coeff, trans_coeff)

    def move(self, dx, dy):
        """
        Moves the lens by dx and dy.

        Args:
            dx (float): The x displacement.
            dy (float): The y displacement.
        """
        self.x += dx
        self.y += dy
        self.update()

    def calculate_points(self):

        self.points = []
        for i in range(len(self.x)):
            self.points.append(np.array(self.x[i], self.y[i]))

    def calculate_segments(self):

        self.segments = []
        for i in range(len(self.x) - 1):
            start = np.array((self.x[i], self.y[i]))
            end = np.array((self.x[i + 1], self.y[i + 1]))
            self.segments.append((start, end))

    def calculate_bounds(self):

        self.bounding_box = []
        self.bounding_box.append(
            np.array(((min(self.x), min(self.y)), (min(self.x), max(self.y))))
        )
        self.bounding_box.append(
            np.array(((min(self.x), min(self.y)), (max(self.x), min(self.y))))
        )
        self.bounding_box.append(
            np.array(((max(self.x), max(self.y)), (min(self.x), max(self.y))))
        )
        self.bounding_box.append(
            np.array(((max(self.x), max(self.y)), (max(self.x), min(self.y))))
        )

    def plot(self, plot_dx=False, plot_bounding_box=False, **kwargs):

        for el in self.segments:
            plt.plot([el[0][0], el[1][0]], [el[0][1], el[1][1]], **kwargs)

        if plot_dx:
            tmp_dx = self.dx - min(self.dx)

            ratio = max(tmp_dx) / max(self.x)

            tmp_dx /= ratio
            for i in range(len(self.y) - 1):
                plt.plot(
                    [tmp_dx[i], tmp_dx[i + 1]],
                    [self.y[i], self.y[i + 1]],
                    alpha=0.2,
                    **kwargs,
                )

        if plot_bounding_box:
            alpha = 0.3
            linestyle = "dashed"
            plt.plot(
                [0, 0],
                [min(self.y), max(self.y)],
                alpha=alpha,
                linestyle=linestyle,
                **kwargs,
            )
            plt.plot(
                [max(self.x), max(self.x)],
                [min(self.y), max(self.y)],
                alpha=alpha,
                linestyle=linestyle,
                **kwargs,
            )
            plt.plot(
                [0, max(self.x)],
                [min(self.y), min(self.y)],
                alpha=alpha,
                linestyle=linestyle,
                **kwargs,
            )
            plt.plot(
                [0, max(self.x)],
                [max(self.y), max(self.y)],
                alpha=alpha,
                linestyle=linestyle,
                **kwargs,
            )
