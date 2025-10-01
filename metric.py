import numpy as np
import scipy

def sample_points_and_face_normals(mesh, sample_count):
  points, indices = mesh.sample(sample_count, return_index=True)
  points = points.astype(np.float32)
  normals = mesh.face_normals[indices]
  return points, normals

def get_points(mesh1, mesh2, points1, points2, sample_count):
  if points1 is not None or points2 is not None:
    assert points1 is not None and points2 is not None
  else:
    points1, _ = sample_points_and_face_normals(mesh1, sample_count)
    points2, _ = sample_points_and_face_normals(mesh2, sample_count)
  return points1, points2

def pointcloud_neighbor_distances_indices(source_points, target_points):
  target_kdtree = scipy.spatial.cKDTree(target_points)
  distances, indices = target_kdtree.query(source_points)
  return distances, indices

def mesh_chamfer_via_points(mesh1,
                            mesh2,
                            sample_count=100000,
                            points1=None,
                            points2=None):
  points1, points2 = get_points(mesh1, mesh2, points1, points2, sample_count)
  dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
  dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
  chamfer = 1000.0 * (np.mean(dist12**2) + np.mean(dist21**2))
  return chamfer

def percent_below(dists, thresh):
  return np.mean((dists**2 <= thresh).astype(np.float32)) * 100.0

def f_score(a_to_b, b_to_a, thresh):
  precision = percent_below(a_to_b, thresh)
  recall = percent_below(b_to_a, thresh)

  return (2 * precision * recall) / (precision + recall + 1e-09)

def fscore(mesh1,
           mesh2,
           sample_count=100000,
           tau=1e-04,
           points1=None,
           points2=None):
  """Computes the F-Score at tau between two meshes."""
  points1, points2 = get_points(mesh1, mesh2, points1, points2, sample_count)
  dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
  dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
  f_score_tau = f_score(dist12, dist21, tau)
  return f_score_tau

def all_mesh_metrics(mesh1, mesh2, sample_count=100000):
  points1, normals1 = sample_points_and_face_normals(mesh1, sample_count)
  points2, normals2 = sample_points_and_face_normals(mesh2, sample_count)

  fs_tau = fscore(mesh1, mesh2, sample_count, 1e-03, points1, points2)
  chamfer = mesh_chamfer_via_points(mesh1, mesh2, sample_count, points1,
                                    points2)
  return fs_tau, chamfer

def all_mesh_metrics_points(points, mesh2, sample_count=100000):
  points1 = points[np.random.choice(points.shape[0], sample_count, replace=False)]
  points2, normals2 = sample_points_and_face_normals(mesh2, sample_count)

  fs_tau = fscore(None, mesh2, sample_count, 1e-03, points1, points2)
  chamfer = mesh_chamfer_via_points(None, mesh2, sample_count, points1,
                                    points2)
  return fs_tau, chamfer