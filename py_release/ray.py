import numpy as np

class Ray:
    def __init__(self, strenght: float, origin: tuple, dir: tuple, color="g"):
        self.strength = strenght
        self.origin = np.array(origin)
        self.dir = np.array(dir)
        self.color = color
        self.end_point = None
        self.length = None

    @classmethod
    def from_ray(cls, r, dir, attenuation=0):
        if r.end_point is None:
            print("Previous ray must be ended before creating a new one")
            return None
        return cls(r.strength * attenuation, r.end_point, dir, r.color)

    def end(self, end_point):
        self.end_point = end_point
        self.length = np.linalg.norm(np.array(self.origin) - np.array(self.end_point))

    def plot(self, plt, color="k", arrow_type="small"):
        if self.end_point is not None:
            plt.plot(
                [self.origin[0], self.end_point[0]],
                [self.origin[1], self.end_point[1]],
                color=color,
                alpha=self.strength,
            )
        else:

            if self.length is None or arrow_type == "small":
                length = 0.1
            else:
                length = self.length / 2

            # draw a small arrow on the origin
            plt.arrow(
                self.origin[0],
                self.origin[1],
                length * self.dir[0],
                length * self.dir[1],
                head_width=0.05,
                head_length=0.05,
                fc=color,
                ec=color,
                alpha=self.strength,
            )
        return

    @staticmethod
    def from_point(start, n, angle_start=0, angle_end=2 * np.pi, color="g"):
        rays = []
        angle = np.linspace(angle_start, angle_end, n)
        for i in range(n):
            rays.append(
                Ray(
                    strenght=1,
                    origin=start,
                    dir=(np.cos(angle[i]), np.sin(angle[i])),
                    color=color,
                )
            )
        return rays

    @staticmethod
    def from_segment(segment, n, direction, color="g"):

        origins = np.linspace(segment[0], segment[1], n)

        rays = []
        for origin in origins:
            rays.append(Ray(strenght=1, origin=origin, dir=direction, color=color))
        return rays

    def __str__(self):
        return f"Strenght: {self.strength}, Origin: {self.origin}, Direction: {self.dir}, End: {self.end_point}, Length: {self.length}"

