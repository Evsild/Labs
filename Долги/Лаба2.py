import random
import matplotlib.pyplot as plt

cluster_points = [[1, 3], [15, 18], [5, 31]]
radii = [3, 4, 7]
x = []
y = []

for cluster_idx in range(3):
    points_count = 0
    while points_count < 20:
        rand_x = random.uniform(cluster_points[cluster_idx][0] - radii[cluster_idx],
                                cluster_points[cluster_idx][0] + radii[cluster_idx])
        rand_y = random.uniform(cluster_points[cluster_idx][1] - radii[cluster_idx],
                                cluster_points[cluster_idx][1] + radii[cluster_idx])
        if (rand_x - cluster_points[cluster_idx][0]) ** 2 + (rand_y - cluster_points[cluster_idx][1]) ** 2 <= radii[
            cluster_idx] ** 2:
            x.append(rand_x)
            y.append(rand_y)
            points_count += 1

points = list(zip(x, y))
num_clusters = 3
centroids = [[random.randint(0, 17), random.randint(0, 40)] for _ in range(num_clusters)]
centroid_x = [c[0] for c in centroids]
centroid_y = [c[1] for c in centroids]

threshold = 999
while threshold > 10 ** (-6):
    dist_matrix = []
    for i in range(num_clusters):
        dist = [[(point[0] - centroid_x[i]) ** 2 + (point[1] - centroid_y[i]) ** 2, i] for point in points]
        dist_matrix.append(dist)

    point_labels = []
    for i in range(len(points)):
        min_dist = min([dist_matrix[j][i] for j in range(num_clusters)])
        point_labels.append(min_dist[1])

    new_centroid_x = []
    new_centroid_y = []
    for cluster_num in range(num_clusters):
        sum_x = 0
        sum_y = 0
        cluster_size = 0
        for i in range(len(point_labels)):
            if point_labels[i] == cluster_num:
                cluster_size += 1
                sum_x += x[i]
                sum_y += y[i]
        if cluster_size != 0:
            sum_x = sum_x / cluster_size
            sum_y = sum_y / cluster_size
        else:
            sum_x = centroid_x[cluster_num]
            sum_y = centroid_y[cluster_num]
        new_centroid_x.append(sum_x)
        new_centroid_y.append(sum_y)

    delta_x = [abs(new_centroid_x[i] - centroid_x[i]) for i in range(len(centroid_x))]
    delta_y = [abs(new_centroid_y[i] - centroid_y[i]) for i in range(len(centroid_y))]
    threshold = max(delta_x + delta_y)
    centroid_x = new_centroid_x
    centroid_y = new_centroid_y

fig, ax = plt.subplots()
for cluster_id in range(num_clusters):
    cluster_points_x = [x[i] for i in range(len(points)) if point_labels[i] == cluster_id]
    cluster_points_y = [y[i] for i in range(len(points)) if point_labels[i] == cluster_id]
    ax.scatter(cluster_points_x, cluster_points_y, label=f'Cluster {cluster_id + 1}')
ax.scatter(centroid_x, centroid_y, c='black', label='Centroids')
ax.legend()
plt.show()