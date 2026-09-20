import numpy as np

def euclid_dist(p1,p2): # p1 - centroid, p2 - point
	p1 = np.array(p1)
	p2 = np.array(p2)
	return p1, np.linalg.norm(p1 - p2)

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	final_centroids = None
	centroids = initial_centroids
	clusters = [[] for _ in range(k)]
	for _ in range(max_iterations):
		for point in points:
			results = [euclid_dist(centroid, point) for centroid in centroids]
			centroid, _ = min(results, key=lambda x: x[1])
			centroid_index = np.where(np.all(centroids == centroid, axis=1))[0][0]
			clusters[centroid_index].append(point)
		centroids = [np.mean(cluster, axis=0) for cluster in clusters]

	final_centroids = list(map(tuple, centroids))

	return final_centroids