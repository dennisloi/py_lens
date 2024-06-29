import numpy as np
import matplotlib.pyplot as plt
from time import time
from lens import Lens
from ray import Ray
from ray_utils import ray_lens_bb_intersect, ray_lens_intersect, reflected_transmitted_rays

MINIMUM_DISTANCE = 1e-6
RAY_STRENGHT_THRESHOLD = 0.1
ITERACTIONS_LIMIT = 10000

test_rays = Ray.from_point(start=(-0.5, 0), n=12, angle_start=-1.2, angle_end=1.2, color='red')
test_rays_seg = Ray.from_segment(segment=((-1, -0.9), (-1, 0.9)), n=10, direction=(1, 0), color='blue')
test_rays_segment = Ray.from_segment(segment=((-0.5, -1), (0, 1)), n=100, direction=(1, 0), color='blue')
test_lens0 = Lens.lens_asphere(res = 100, R = 1, k = -1, a = [0], refidx_l=1, refidx_r=1.5, refl_coeff=0.3)
test_lens1 = Lens.lens_asphere(res = 100, R = 1, k = -0.1, a = [0], refidx_l=1.5, refidx_r=1, refl_coeff=0.3)
test_seg_lens = Lens.lens_segment(res = 100, start = (1, -1), end = (-1, 1), refidx_l=1, refidx_r=1.5, refl_coeff=0.3)
test_seg_lens2 = Lens.lens_segment(res = 100, start = (1, -1), end = (-1, 1), refidx_l=1, refidx_r=1.5, refl_coeff=0.3)
test_seg_lens2.move(0.3, 0)
test_lens1.x += 0.2
test_lens1.update()

rays_pool = [Ray(1, (1, 0.2), (-1, 0))]
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
        reflected_ray = Ray.from_ray(current_ray, refl_dir, intersected_lens.refl_coeff)
        transmitted_ray = Ray.from_ray(current_ray, transm_dir, intersected_lens.trans_coeff)

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