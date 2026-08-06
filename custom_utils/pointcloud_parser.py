

def get_pcd_from_colmap(self):
    points = []
    colors = []
    for p in self.recon.points3D.values():
        points.append(p.xyz)
        colors.append(p.color / 255.0)
    points = np.asarray(points)
    colors = np.asarray(colors)

    return points, colors

def get_pcd_from_ply(self, pcd_path):
    g = trimesh.load(pcd_path, process=False)
    if isinstance(g, trimesh.PointCloud):
        points = np.asarray(g.vertices, dtype=np.float32)
        colors = getattr(g, "colors", None)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.uint8)[:, :3]
    else:
        points = np.asarray(g.vertices, dtype=np.float32)
        colors = None
    
    return points, colors

def get_pcd_from_npy(self, pcd_path):
    points = np.load(pcd_path)
    colors = np.repeat([[0., 0., 1.0]], len(points), axis=0)

    return points, colors

def get_pcd_from_torch(self, pcd_path):
    points = torch.load(pcd_path)
    pts = points[:, :3].numpy()
    cols = points[:, 3:6].numpy()

    return pts, cols

def get_pcd_from_mesh_obj(self, pcd_path):
    mesh = trimesh.load(pcd_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    
    points, face_idx = trimesh.sample.sample_surface(mesh, 100000)
    colors = mesh.visual.face_colors[face_idx][:, :3] # remove alpha

    print(type(points))
    print(points.shape)
    print(colors.shape)

    return points, colors

def get_pcd_from_pcd(self, pcd_path):
    pcd_o3d = o3d.io.read_point_cloud(pcd_path)
    
    points = np.asarray(pcd_o3d.points)
    colors = np.asarray(pcd_o3d.colors) # 0~1 사이 값

    P = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float64)
    points = (P @ points.T).T
    
    # g = trimesh.points.PointCloud(vertices=points, colors=colors)
    return points, colors