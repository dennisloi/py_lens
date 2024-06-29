from matplotlib import pyplot as plt
import numpy as np
from lens import Lens


class Ray:
    """
    Represents a ray of light in a 2D space.

    Attributes:
        strength (float): The strength of the ray.
        origin (tuple): The origin point of the ray.
        direction (tuple): The direction vector of the ray.
        color (str): The color of the ray.
        end_point: The end point of the ray.
        length: The length of the ray.
    """

    def __init__(self, strength: float, origin: tuple, direction: tuple, color="g"):
        self.strength = strength
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.color = color
        self.end_point = None
        self.length = None

    @classmethod
    def from_ray(cls, r, direction, attenuation=0):
        """
        Creates a new ray from an existing ray.

        Args:
            r (Ray): The previous ray.
            direction (tuple): The new direction of the ray.
            attenuation (float): The attenuation of the ray.

        Returns:
            Ray: The new ray.
        """
        if r.end_point is None:
            print("Previous ray must be ended before creating a new one")
            return None
        return cls(r.strength * (1 - attenuation), r.end_point, direction, r.color)

    @staticmethod
    def from_point(start, n, angle_start=0, angle_end=2 * np.pi, color="g"):
        """
        Creates a set of rays from a point.

        Args:
            start (tuple): The starting point of the rays.
            n (int): The number of rays.
            angle_start (float): The starting angle of the rays.
            angle_end (float): The ending angle of the rays.
            color (str): The color of the rays.

        Returns:
            list: The list of rays.
        """
        angle = np.linspace(angle_start, angle_end, n)
        rays = [
            Ray(
                strength=1,
                origin=start,
                direction=(np.cos(angle[i]), np.sin(angle[i])),
                color=color,
            )
            for i in range(n)
        ]
        return rays

    @staticmethod
    def from_segment(segment, n, direction, color="g"):
        """
        Creates a set of rays from a segment.

        Args:
            segment (tuple): The segment.
            n (int): The number of rays.
            direction (tuple): The direction of the rays.
            color (str): The color of the rays.
        
        Returns:
            list: The list of rays.
        """
        origins = np.linspace(segment[0], segment[1], n)
        rays = [
            (Ray(strength=1, origin=origin, direction=direction, color=color))
            for origin in origins
        ]

        return rays

    def __str__(self):
        return f"strength: {self.strength}, Origin: {self.origin}, Direction: {self.direction}, End: {self.end_point}, Length: {self.length}"

    def end(self, end_point):
        self.end_point = end_point
        self.length = np.linalg.norm(np.array(self.origin) - np.array(self.end_point))

def plot(r: Ray, color="k", arrow_type="small"):

    """
    Plots the ray on a matplotlib figure.

    Args:
        plt: The matplotlib figure.
        color (str): The color of the ray.
        arrow_type (str): The type of arrow to plot.
    """
    if r.end_point is not None:
        plt.plot(
            [r.origin[0], r.end_point[0]],
            [r.origin[1], r.end_point[1]],
            color=color,
            alpha=r.strength,
        )
    else:

        if r.length is None or arrow_type == "small":
            length = 0.1
        else:
            length = r.length / 2

        # draw a small arrow on the origin
        plt.arrow(
            r.origin[0],
            r.origin[1],
            length * r.direction[0],
            length * r.direction[1],
            head_width=0.05,
            head_length=0.05,
            fc=color,
            ec=color,
            alpha=r.strength,
        )


def ray_segment_intersect(r: Ray, s: np.array):
    # Ray
    o1 = r.origin
    d1 = r.direction
    # Segment
    o2 = s[0]
    d2 = s[1] - s[0]

    # Calculate intersection point
    cross = np.cross(d1, d2)

    # If the cross product is zero, the lines are parallel
    if cross == 0:
        return None

    t1 = np.cross(o2 - o1, d2) / cross
    t2 = np.cross(o2 - o1, d1) / cross

    if t1 >= 0 and 0 <= t2 <= 1:
        return o1 + t1 * d1

    return None


def ray_lens_bb_intersect(r: Ray, l: Lens):
    # Checks if a ray intersects with a lens' bounding box

    for segment in l.bounding_box:
        intersection = ray_segment_intersect(r, segment)
        if intersection is not None:
            return intersection

    return None


def ray_lens_intersect(r: Ray, l: Lens):
    # Checks if a ray intersects with a lens

    hit = False
    dist_min = np.inf

    for i, segment in enumerate(l.segments):
        intersection = ray_segment_intersect(r, segment)

        if intersection is not None:
            hit = True
            dist = np.linalg.norm(r.origin - intersection)
            if dist < dist_min:
                dist_min = dist
                intersected_segment = segment
                intersected_point = intersection
                index = i

    if hit:
        return intersected_segment, intersected_point, index

    return None, None, None


def interpolate_dx(l: Lens, intersection, segment, index):
    """Calculates the distance between the two points of the intersected segment
    Then interpolates the value of dx at the intersection point

    Args:
        l (Lens): The lens object
        intersection (np.array): The intersection point
        segment (np.array): The segment points
        index (int): The index of the segment

    Returns:
        float: The interpolated value of dx at the intersection point
    """

    # Points of the segment
    p1, p2 = np.array(segment[0]), np.array(segment[1])

    # Distances between the intersection point and the segment points
    d1, d2 = np.linalg.norm(p1 - intersection), np.linalg.norm(p2 - intersection)

    # Interpolated value of dx over the segment
    dx = l.dx[index] * (d2 / (d1 + d2)) + l.dx[index + 1] * (d1 / (d1 + d2))

    return dx


def reflected_transmitted_rays(r: Ray, l: Lens, p: np.array, s: np.array, index):
    # Get the dx in the given point
    dx = interpolate_dx(l, s, p, index)

    # Calculate the normal of the lense from the dx
    n = -np.array([1, dx])
    n = n / np.linalg.norm(n)

    # Calculate the parallel component of the ray in respect to the normal
    r_parallel = np.dot(r.direction, n) * n

    # Calculate the perpendicular component of the ray in respect to the normal
    r_perpend = r.direction - r_parallel

    n1 = l.refidx_l
    n2 = l.refidx_r

    # Source: https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
    transmitted_dir = -np.sqrt(
        1 - (n1 / n2) ** 2 * (1 - np.dot(n, r.direction) ** 2)
    ) * n + (n1 / n2) * (r.direction - np.dot(n, r.direction) * n)

    reflected_ray = r_perpend - r_parallel

    return reflected_ray, transmitted_dir
