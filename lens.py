import numpy as np
import matplotlib.pyplot as plt
import time

MINIMUM_DISTANCE = 0.1
RAY_STRENGHT_THRESHOLD = 0.1
ITERACTIONS_LIMIT = 10000

class ray:
    def __init__(self, strenght : float, origin : tuple, dir : tuple, color = 'g'):
        self.strength = strenght
        self.origin = np.array(origin)
        self.dir = np.array(dir)
        self.color = color
        self.end_point = None
        self.length = None

    @classmethod
    def from_ray(cls, r, dir, attenuation = 0):
        if r.end_point is None:
            print("Previous ray must be ended before creating a new one")
            return None
        return cls(r.strength * attenuation, r.end_point, dir, r.color)

    def end(self, end_point):
        self.end_point = end_point
        self.length = np.linalg.norm(np.array(self.origin) - np.array(self.end_point))

    def plot(self, plt, color = 'k', arrow_type = 'small'):
        if self.end_point is not None:
            plt.plot([self.origin[0], self.end_point[0]], [self.origin[1], self.end_point[1]], color=color, alpha=self.strength)
        else:
        

            if self.length is None or arrow_type == 'small':
                length = 0.1
            else:
                length = self.length / 2
            
            #draw a small arrow on the origin
            plt.arrow(self.origin[0], self.origin[1], length*self.dir[0], length*self.dir[1], head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=alpha)
        return
        
    @staticmethod
    def from_point(start, n, angle_start=0, angle_end=2*np.pi, color='g'):
        rays = []
        angle = np.linspace(angle_start, angle_end, n)
        for i in range(n):
            rays.append(ray(strenght=1, origin=start, dir=(np.cos(angle[i]), np.sin(angle[i])), color=color))
        return rays
    
    @staticmethod
    def from_segment(segment, n, direction, color='g'):

        origins = np.linspace(segment[0], segment[1], n)

        rays = []
        for origin in origins:
            rays.append(ray(strenght=1, origin=origin, dir=direction, color=color))
        return rays

    
    def __str__(self):
        return f"Strenght: {self.strength}, Origin: {self.origin}, Direction: {self.dir}, End: {self.end_point}, Length: {self.length}"

class lens:
    def __init__(self, x : np.array, y : np.array, dx : np.array, refidx_l : float = 1, refidx_r : float = 1, refl_coeff : float = 0.1, trans_coeff : float = 0.9):

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
        self.calculate_segments()
        self.calculate_points()
        self.calculate_bounds()

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.update()

    def calculate_points(self):

        self.points = []
        for i in range(len(self.x)):
            self.points.append(np.array(self.x[i], self.y[i]))

    def calculate_segments(self):

        self.segments = []
        for i in range(len(self.x)-1):
            start = np.array((self.x[i], self.y[i]))
            end = np.array((self.x[i+1], self.y[i+1]))
            self.segments.append((start, end))

    def calculate_bounds(self):

        self.bounding_box = []
        self.bounding_box.append(np.array(((min(self.x), min(self.y)), (min(self.x), max(self.y)))))
        self.bounding_box.append(np.array(((min(self.x), min(self.y)), (max(self.x), min(self.y)))))
        self.bounding_box.append(np.array(((max(self.x), max(self.y)), (min(self.x), max(self.y)))))
        self.bounding_box.append(np.array(((max(self.x), max(self.y)), (max(self.x), min(self.y)))))   

    @classmethod
    def lens_asphere(cls, res : int, R : float, k : float, a : list, refidx_l : float = 1, refidx_r : float = 1, refl_coeff : float = .01, trans_coeff : float = 0.9):

        # Sample the -1 to 1 range with equidistant points
        y = np.linspace(-1, 1, res)

        sqrt_el = np.sqrt(1 - ((1 + k)*y**2)/R**2)

        # Aspheric lense formula
        x = y**2 / (R * (1 + sqrt_el)) -5

        # Additional terms
        for i in range(len(a)):
            x += a[i] * y**(4 + i*2)

        # Derivative
        #https://www.wolframalpha.com/input?i=derivative+of+x%28y%29+%3D+y%5E2+%2F+%28R+%281+%2B+sqrt%281+-+%28%281+%2B+k%29+y%5E2%29+%2F+R%5E2%29%29%29+%2B+a4+y%5E4+%2B+a6+y%5E6
        dx = 2*y / (R*(sqrt_el + 1)) + ((k + 1) * y**3) / (R**3*(sqrt_el)*(sqrt_el + 1)**2)

        # Additional terms
        for i in range(len(a)):
            dx += (4 + i*2) * a[i] * y**(4 + i*2 - 1)

        dx = - dx
        
        # Move the lense to the origin
        x -= min(x)

        return cls(x, y, dx, refidx_l, refidx_r, refl_coeff, trans_coeff)

    @classmethod
    def lens_segment(cls, res : int, start : np.array, end : np.array, refidx_l : float = 1, refidx_r : float = 1, refl_coeff : float = .01, trans_coeff : float = 0.9):

        x = np.linspace(start[0], end[0], res)
        y = np.linspace(start[1], end[1], res)
        if end[1] == start [1]:
            if start[0] > end[0]:
                dx = np.array(res*np.inf)
            else:
                dx = -np.array(res*np.inf)
        else:
            dx = np.array(res* [ (start[0]-end[0])/(end[1]-start[1])])

        return cls(x, y, dx, refidx_l, refidx_r, refl_coeff, trans_coeff)


    def plot(self, plt, plot_dx = False, plot_bounding_box = False, **kwargs):

        for el in self.segments:
            plt.plot([el[0][0], el[1][0]], [el[0][1], el[1][1]], **kwargs)

        

        if plot_dx:
            tmp_dx = self.dx - min(self.dx)

            ratio = max(tmp_dx) / max(self.x)

            tmp_dx  /= ratio
            for i in range(len(self.y)-1):
                plt.plot([tmp_dx[i], tmp_dx[i+1]], [self.y[i], self.y[i+1]], alpha = 0.2, **kwargs)

        if plot_bounding_box:
            alpha = 0.3
            linestyle='dashed'
            plt.plot([0, 0], [min(self.y), max(self.y)], alpha = alpha, linestyle = linestyle, **kwargs)
            plt.plot([max(self.x), max(self.x)], [min(self.y), max(self.y)], alpha = alpha, linestyle = linestyle, **kwargs)
            plt.plot([0, max(self.x)], [min(self.y), min(self.y)], alpha = alpha, linestyle = linestyle, **kwargs)
            plt.plot([0, max(self.x)], [max(self.y), max(self.y)], alpha = alpha, linestyle = linestyle, **kwargs)

def ray_segment_intersect(r : ray, s : np.array):
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

def ray_lens_bb_intersect(r : ray, l : lens):    
    # Checks if a ray intersects with a lens' bounding box
    
    for segment in l.bounding_box:
        intersection = ray_segment_intersect(r, segment)
        if intersection is not None:
            return intersection
        
    return None

def ray_lens_intersect(r: ray, l: lens):
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

def interpolate_dx(l : lens, intersection, segment, index):
    
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

def reflected_transmitted_rays(r : ray, l : lens, p : np.array, s : np.array, index):

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
    transmitted_dir = - np.sqrt(1 - (n1/n2)**2 * (1 - np.dot(n, r.dir)**2)) * n + (n1/n2) * (r.dir - np.dot(n, r.dir)*n)

    reflected_ray = r_perpend - r_parallel


    return reflected_ray, transmitted_dir



test_rays = ray.from_point(start=(-0.5, 0), n=12, angle_start=-1.2, angle_end=1.2, color='red')
test_rays_seg = ray.from_segment(segment=((-1, -0.9), (-1, 0.9)), n=10, direction=(1, 0), color='blue')
test_rays_segment = ray.from_segment(segment=((-0.5, -1), (0, 1)), n=100, direction=(1, 0), color='blue')
test_lens0 = lens.lens_asphere(res = 100, R = 1, k = -1, a = [0], refidx_l=1, refidx_r=1.5, refl_coeff=0.3)
test_lens1 = lens.lens_asphere(res = 100, R = 1, k = -0.1, a = [0], refidx_l=1.5, refidx_r=1, refl_coeff=0.3)
test_seg_lens = lens.lens_segment(res = 100, start = (1, -1), end = (-1, 1), refidx_l=1, refidx_r=1.5, refl_coeff=0.3)
test_seg_lens2 = lens.lens_segment(res = 100, start = (1, -1), end = (-1, 1), refidx_l=1, refidx_r=1.5, refl_coeff=0.3)
test_seg_lens2.move(0.3, 0)
test_lens1.x += 0.2
test_lens1.update()

rays_pool = [ray(1, (1, 0.2), (-1, 0))]
#rays_pool = test_rays_seg
rays_done = []

lenses = [test_seg_lens2]
print(test_lens0.dx)

iterations = 0

start_time = time.time()

while len(rays_pool) > 0:

    # Pick the first ray of the pool
    current_ray = rays_pool.pop(0)

    # Check which lens can have an intersection using the bounding box (fast)

    possible_lenses = []

    for lens in lenses:
        if ray_lens_bb_intersect(current_ray, lens) is not None:
            possible_lenses.append(lens)

    # From the possible lenses, calculate the actual intersections to find the closest

    dist_min = np.inf
    hit = False

    for lens in possible_lenses:
        segment, intersection, index = ray_lens_intersect(current_ray, lens)
        if intersection is not None:
            dist = np.linalg.norm(current_ray.origin - intersection)
            if dist < dist_min and dist > MINIMUM_DISTANCE:
                hit = True
                dist_min = dist
                intersected_segment = segment
                intersected_point = intersection
                intersected_lens = lens
                intersected_index = index
    if hit == True:
        # Close the current ray
        current_ray.end(intersected_point)
        rays_done.append(current_ray)

        # Calculate the reflected and transmitted rays
        refl_dir, transm_dir = reflected_transmitted_rays(current_ray, intersected_lens, intersected_point, intersected_segment, intersected_index)

        # Create the rays
        reflected_ray = ray.from_ray(current_ray, refl_dir, intersected_lens.refl_coeff)
        transmitted_ray = ray.from_ray(current_ray, transm_dir, intersected_lens.trans_coeff)

        # Add the rays to the pool
        if reflected_ray.strength > RAY_STRENGHT_THRESHOLD:
            rays_pool.append(reflected_ray)
        if transmitted_ray.strength > RAY_STRENGHT_THRESHOLD:
            rays_pool.append(transmitted_ray)
        
    else:
        #TODO, maybe terminate the ray at the limit of the current window? or inf (idk if possible)?
        current_ray.end(current_ray.origin + current_ray.dir * 10)
        rays_done.append(current_ray)
        continue

    if iterations >= ITERACTIONS_LIMIT:
        print("Iteractions limit reached")
        break
    else:
        iterations += 1
print(f'Rendering time: {time.time() - start_time}')
print(f'Iteractions: {iterations}')
print(f'Rays:{len(rays_done)}')

# Initiate the plot
plt.figure()
plt.xlim(-1, 1.5)
plt.ylim(-1, 1)

# Plot the lenses
for lens in lenses:
    lens.plot(plt, color = 'red')

# Plot the rays
for ray in rays_done:
    ray.plot(plt)

plt.grid()
plt.show()