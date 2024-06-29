import numpy as np
from py_release.lens import Lens
from py_release.ray import Ray


def ray_segment_intersect(r: Ray, s: np.array):
    # Ray
    o1 = r.origin
    d1 = r.dir
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

    if t1 >= 0 and t2 >= 0 and t2 <= 1:
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
    else:
        return None, None, None


def interpolate_dx(l: Lens, intersection, segment, index):

    # Calculates the distance between the two points of the intersected segment
    # Then interpolates the value of dx at the intersection point
    p1 = np.array(segment[0])
    p2 = np.array(segment[1])

    # Distances between the intersection point and the segment points
    d1 = np.linalg.norm(p1 - intersection)
    d2 = np.linalg.norm(p2 - intersection)

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
    r_parallel = np.dot(r.dir, n) * n

    # Calculate the perpendicular component of the ray in respect to the normal
    r_perpend = r.dir - r_parallel

    n1 = l.refidx_l
    n2 = l.refidx_r

    # Source: https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
    transmitted_dir = -np.sqrt(1 - (n1 / n2) ** 2 * (1 - np.dot(n, r.dir) ** 2)) * n + (
        n1 / n2
    ) * (r.dir - np.dot(n, r.dir) * n)

    reflected_ray = r_perpend - r_parallel

    return reflected_ray, transmitted_dir