import numpy as np
from scipy.spatial import distance

class PolygonCluster:
    def __init__(self, threshold):
        self.polygons = []
        self.threshold = threshold

    def add_polygon(self, label, coordinates):
        self.polygons.append({'label': label, 'coordinates': coordinates})

    def calculate_centroid(self, polygon):
        x = [point[0] for point in polygon]
        y = [point[1] for point in polygon]
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        return centroid_x, centroid_y

    def distance_between_polygons(self, polygon1, polygon2):
        centroid1 = self.calculate_centroid(polygon1)
        centroid2 = self.calculate_centroid(polygon2)
        return distance.euclidean(centroid1, centroid2)

    def group_polygons_into_clusters(self):
        clusters = []
        for polygon in self.polygons:
            new_cluster = True
            for cluster in clusters:
                centroid = self.calculate_centroid(cluster[0]['coordinates'])
                dist = self.distance_between_polygons(polygon['coordinates'], cluster[0]['coordinates'])
                if dist <= self.threshold:
                    cluster.append(polygon)
                    new_cluster = False
                    break
            if new_cluster:
                clusters.append([polygon])
        return clusters

# Example usage:
polygon_data = [
    {'label': 'A', 'coordinates': [(1, 1), (1, 2), (2, 2), (2, 1)]},
    {'label': 'B', 'coordinates': [(4, 4), (4, 5), (5, 5), (5, 4)]},
    {'label': 'C', 'coordinates': [(10, 10), (10, 11), (11, 11), (11, 10)]},
    {'label': 'D', 'coordinates': [(20, 20), (20, 21), (21, 21), (21, 20)]},
    {'label': 'E', 'coordinates': [(25, 25), (25, 26), (26, 26), (26, 25)]},
]

threshold = 5  # Adjust the threshold as needed

pc = PolygonCluster(threshold)
for polygon in polygon_data:
    pc.add_polygon(polygon['label'], polygon['coordinates'])

clusters = pc.group_polygons_into_clusters()
for i, cluster in enumerate(clusters):
    print(f'Cluster {i+1}: {[polygon["label"] for polygon in cluster]}')




import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

class PolygonCluster:
    def __init__(self, polygons):
        self.polygons = polygons
        self.centroids = self.calculate_centroids()
        self.labels = self.cluster_polygons()

    def calculate_centroids(self):
        centroids = []
        for polygon in self.polygons:
            centroid = np.mean(polygon, axis=0)
            centroids.append(centroid)
        return np.array(centroids)

    def cluster_polygons(self):
        epsilon = 0.5  # Adjust this threshold as needed
        db = DBSCAN(eps=epsilon, min_samples=1).fit(self.centroids)
        labels = db.labels_
        return labels

    def group_clusters(self):
        clusters = defaultdict(list)
        for i, label in enumerate(self.labels):
            clusters[label].append(self.polygons[i])
        return clusters

# Example usage:
polygons = [
    np.array([[1, 2], [2, 3], [3, 4]]),
    np.array([[5, 6], [6, 7], [7, 8]]),
    np.array([[10, 12], [12, 13], [13, 14]]),
    np.array([[15, 16], [16, 17], [17, 18]]),
    np.array([[20, 22], [22, 23], [23, 24]]),
    np.array([[25, 26], [26, 27], [27, 28]])
]

polygon_cluster = PolygonCluster(polygons)
clusters = polygon_cluster.group_clusters()

for cluster_id, polygons in clusters.items():
    print(f"Cluster {cluster_id}:")
    for polygon in polygons:
        print(polygon)

