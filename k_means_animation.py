import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters for simulated data
n = 1000
center1 = (1, 5)
center2 = (6, 2)

# Simulate normally-distributed data
rng = np.random.default_rng()
data1 = rng.multivariate_normal(mean=np.array(center1), 
                               cov=np.array([[1, 0.5],[0.5, 1.5]]),
                               size=n//2)
data2 = rng.multivariate_normal(mean=np.array(center2), 
                               cov=np.array([[1.25, 0],[0, 0.75]]),
                               size=n//2)
data = np.vstack((data1, data2))


def k_means_cluster(data, k=2, tol=10e-3):
    """Performs k-means clustering on given data (nxd where n is number of samples) for k
    clusters. Algorithm is halted when calculated means are within tol of the nearest
    possible means. Returns an array containining the calculated means and the cluster
    (numbered 0...(k-1)) each sample belongs to for each iteration for animation purposes."""
    
    # Randomly choose first means from the data
    rng = np.random.default_rng()
    first_means_idx = rng.integers(data.shape[0], size=k)
    means = [data[first_means_idx]]
    clusters = []
    
    change = 10e6
    while change > tol:
        # Calculate distance from each data point to each mean
        distances = np.zeros((n, k))
        for i in range(k):
            distances[:,i] = np.linalg.vector_norm(data - means[-1][i], axis=1)
        
        # Cluster by minimum distance (column index of distances is cluster number)
        new_clusters = np.argmin(distances, axis=1)
        clusters.append(new_clusters)
        
        # Calculate new means as average position of each cluster
        new_means = np.zeros((k, data.shape[1]))
        for i in range(k):
            new_means[i, :] = np.average(data[new_clusters == i], axis=0)
        means.append(new_means)
        
        # Calculate change from previous means and repeat
        change = np.sum(np.linalg.vector_norm(means[-1]-means[-2], axis=0))
    
    # Do one more distance calculation to get the final clustering
    distances = np.zeros((n, k))
    for i in range(k):
        distances[:,i] = np.linalg.vector_norm(data - means[-1][i], axis=1)
    new_clusters = np.argmin(distances, axis=1)
    clusters.append(new_clusters)
    
    return np.array(means), np.array(clusters)

k=2
k_means_results, k_means_clusters = k_means_cluster(data, k)

# Plot initial set-up of problem
fig, ax = plt.subplots()
ax.scatter(x=data[:,0], y=data[:,1])
ax.set_title("Initial Set-Up")
fig.savefig("k_means_initial.png")
#plt.show()

# Animate each iteration
fig, ax = plt.subplots()
colors = ["blue", "green"]
def update(i):
    ax.clear()
    if i == 0:
        ax.text(0.5, 1.05, "k-means clustering, Initial Data", 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes)
        ax.scatter(x=data[:,0], y=data[:,1], color="teal", label="Original Data")
        ax.scatter(x=[center1[0], center2[0]], y=[center1[1], center2[1]],
                                color="orange", label="True Center")
    else:
        ax.text(0.5, 1.05, f"k-means clustering, Iteration {i}", 
                            size=plt.rcParams["axes.titlesize"],
                            ha="center", transform=ax.transAxes)
        ax.scatter(x=[center1[0], center2[0]], y=[center1[1], center2[1]],
                                        color="orange", label="True Center", zorder=1)
        ax.scatter(x=k_means_results[i-1,:,0], y=k_means_results[i-1,:,1],
                                        color = "red", label="Calculated Centers", zorder=1)
        for j in range(k):
            ax.scatter(x=data[k_means_clusters[i-1] == j, 0],
                                    y=data[k_means_clusters[i-1] == j, 1],
                                    color=colors[j], label=f"Cluster {j}", zorder=0)
        
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=k_means_results.shape[0]+1,
                              interval = 3000, repeat_delay=1000)
ani.save('k_means_animation.gif')
#plt.show()

# Plot correct answer vs calculated answer
fig, axs = plt.subplots(1, 2, figsize=(8,5))
axs[0].set_title("Correct clusters")
axs[1].set_title("Calculated clusters")
axs[0].scatter(x=data1[:,0], y=data1[:,1], color="blue", label="Cluster 1")
axs[0].scatter(x=data2[:,0], y=data2[:,1], color="green", label="Cluster 2")
axs[1].scatter(x=data[k_means_clusters[-1] == 0, 0],
                                    y=data[k_means_clusters[-1] == 0, 1],
                                    color="blue", label=f"Cluster {1}")
axs[1].scatter(x=data[k_means_clusters[-1] == 1, 0],
                                    y=data[k_means_clusters[-1] == 1, 1],
                                    color="green", label=f"Cluster {2}")
axs[0].legend()
axs[1].legend()
fig.tight_layout()
fig.savefig("k_means_correct.png")
#plt.show()

# Half-moon data set
def upper_moon(x):
    return np.sqrt(1-(x-1)**2)+1

def lower_moon(x):
    return -np.sqrt(1-(x-2)**2)+1.5

x_upper = rng.uniform(0, 2, n//2)
y_upper = upper_moon(x_upper)
x_lower = rng.uniform(1, 3, n//2)
y_lower = lower_moon(x_lower)
half_moon_data = np.hstack( (np.vstack((x_upper, y_upper)), np.vstack((x_lower, y_lower)) ) )
half_moon_data = np.transpose(half_moon_data)
half_moon_data += rng.normal(loc=0, scale=0.15, size=half_moon_data.shape)

half_moon_results, half_moon_clusters = k_means_cluster(half_moon_data, k)

fig, ax = plt.subplots()
ax.scatter(x=half_moon_data[half_moon_clusters[-1] == 0, 0],
                y=half_moon_data[half_moon_clusters[-1] == 0, 1],
                color="blue", label="Cluster 1")
ax.scatter(x=half_moon_data[half_moon_clusters[-1] == 1, 0],
                y=half_moon_data[half_moon_clusters[-1] == 1, 1],
                color="green", label="Cluster 2")
ax.set_title("Half-moon data")
ax.legend()
fig.savefig("k_means_half_moon.png")