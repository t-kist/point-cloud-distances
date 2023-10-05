import numpy as np
import pickle, gzip
import pylab as plt
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan
import jax
import gc
import os
import sys
import gzip
import wget
from ott.geometry import segment, pointcloud, geometry
from ott.problems.quadratic import gw_barycenter as gw_prob
from ott.solvers.quadratic import gw_barycenter as gw_solver
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from ott.solvers.linear import sinkhorn
from IPython import embed
import time

FILE = 'halos.pkl.gz'
URL = 'https://zenodo.org/record/8245981/files/' + FILE

def read_data(v_scale=8.0): # magic 8
    if not os.path.isfile(FILE):
        fn = wget.download(URL)
        print(fn, FILE)
    with gzip.open(FILE, "r") as f:
        particles, halo_slice = pickle.load(f)
    # TEMPORARY FIX for bad data -- RUN THIS ONLY ONCE
    halo_slice['GroupStellarMass'] = halo_slice['GroupStellarMass'][halo_slice['GroupID']]
    label = np.log10(halo_slice['GroupStellarMass'])

    for i, id in enumerate(halo_slice['GroupID']):
        R200 = halo_slice['Group_R_Mean200'][i]
        G = 4.30219372e10 / 1e6
        V200 = np.sqrt(G * halo_slice['Group_M_Mean200'][i] / R200)
        # print(i, "halo", id, "R200", R200, "V200", V200)
        particles[i]['xs'] = np.concatenate((particles[i]["Coordinates"] / R200,
                                            particles[i]["Velocities"] / V200 / v_scale), #magic number
                                            axis=1)
    for p in particles:
        p['xs'] = p['xs'] - np.mean(p['xs'], axis=0)
    return particles,label

def subsample(x, k):
    n = x.shape[0]
    if n < k:
        return x
    aux = np.random.choice(range(n), size=k, replace=False)
    return x[aux, :]

def dist_function(xx, yy, solver, wxx=None, wyy=None):
    """
    Compute the Gromov-Wasserstein distances between one set of point clouds and another.
    """
    geom_xx = pointcloud.PointCloud(xx)
    geom_yy = pointcloud.PointCloud(yy)
    out = solver(quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a=wxx, b=wyy, scale_cost=False))
    return out.primal_cost, out.converged, out.linear_convergence

def experiments(a_size, b_size, num_points, v_scale, epsilon, iterations):
    particles, label = read_data(v_scale)
    np.random.seed(42)
    rand_label = np.random.uniform(size=len(label))
    a_index = np.argsort(rand_label)[:a_size]
    b_index = np.argsort(rand_label)[-b_size:]
    a_point_clouds = [subsample(particles[i]['xs'], num_points) for i in a_index]
    b_point_clouds = [subsample(particles[i]['xs'], num_points) for i in b_index]
    a_labels = label[a_index]
    b_labels = label[b_index]
    a_sizes = [point.shape[0] for point in a_point_clouds]
    b_sizes = [point.shape[0] for point in b_point_clouds]
    a_train_points, a_train_weights = segment.segment_point_cloud(jnp.concatenate(a_point_clouds), num_per_segment=a_sizes)
    b_train_points, b_train_weights = segment.segment_point_cloud(jnp.concatenate(b_point_clouds), num_per_segment=b_sizes)

    #freeing up memory
    del particles
    del label
    gc.collect()

    d_solver = gromov_wasserstein.GromovWasserstein(min_iterations=iterations, max_iterations=iterations, epsilon=epsilon, linear_ot_solver=sinkhorn.Sinkhorn(min_iterations=1000, max_iterations=1000))

    t = time.time()
    def scanable_inner_loop(ii, xx):
        xxs, wwx = xx
        return ii+1, inner_loop(xxs, b_train_points, d_solver, wwx, b_train_weights)

    ii, (kernel_matrix, converged1, linear_conv1) = scan(scanable_inner_loop, 0, [a_train_points, a_train_weights])
    print(kernel_matrix)
    elapsed = time.time() - t
    print("elapsed scan", elapsed)

    #t = time.time()
    #kernel_matrix, converged, linear_conv = vectorized_dist_function_double_loop(a_train_points, b_train_points, a_train_weights, b_train_weights, d_solver)
    #print(kernel_matrix)
    #elapsed = time.time() - t
    #print("elapsed vmapx2", elapsed)

    label_difference = a_labels[:, None] - b_labels[None,:]

    plt.clf()
    plt.scatter(kernel_matrix.flatten(), label_difference.flatten()**2, marker='.')
    plt.semilogx()
    plt.xlim(0.9*kernel_matrix.min(), 90*kernel_matrix.min())
    plt.savefig("plot_numpoints{0}_epsilon{1}_vscale{2}_iterations{3}.pdf".format(num_points, epsilon, v_scale, iterations))

    pickle.dump((kernel_matrix, label_difference, elapsed), open("numpoints{0}_epsilon{1}_vscale{2}_iterations{3}.pkl".format(num_points, epsilon, v_scale, iterations), "wb"))
    print("numpoints{0}_epsilon{1}_vscale{2}_iterations{3}.pkl".format(num_points, epsilon, v_scale, iterations))



a_size = 13
b_size = 17
num_points = [100, ]
v_scales = [8, ]
epsilons = [0.001, ]
iterations = [20, ] # experiments suggest that 20 is okay, it'd be nice to see if it converged



vectorized_dist_function_single_loop = jax.jit(vmap(dist_function, (0, 0, None, 0, 0), 0))
inner_loop = vmap(dist_function, (None, 0, None, None, 0), 0)
vectorized_dist_function_double_loop = jax.jit(vmap(inner_loop, (0, None, None, None, 0), 0))

for n in num_points:
    for eps in epsilons:
        for v in v_scales:
            for it in iterations:
                experiments(a_size, b_size, n, v, eps, it)
