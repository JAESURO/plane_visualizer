import open3d as o3d
import numpy as np

MODEL_PATH = "the-world-trade-center/source/The World Trade Center.obj"

print("=" * 80)
print("Assignment #5: 3D Model Processing with Open3D")
print("=" * 80)
print()

print("STEP 1: Loading and Visualization")
print("-" * 80)

print(f"Loading model from: {MODEL_PATH}")
mesh = o3d.io.read_triangle_mesh(MODEL_PATH)

print(f"Number of vertices: {len(mesh.vertices)}")
print(f"Number of triangles: {len(mesh.triangles)}")
print(f"Has color: {mesh.has_vertex_colors()}")
print(f"Has normals: {mesh.has_vertex_normals()}")

mesh.compute_vertex_normals()

print("\nDisplaying original model...")
o3d.visualization.draw_geometries([mesh], window_name="Step 1: Original Model")

print("\nUnderstanding: Loaded 3D model in .obj format. The model is a mesh consisting of vertices and triangles that form the surface of the object.\n")

print("STEP 2: Conversion to Point Cloud")
print("-" * 80)

temp_pcd_path = "temp_point_cloud.pcd"
point_cloud_from_mesh = mesh.sample_points_uniformly(number_of_points=100000)
o3d.io.write_point_cloud(temp_pcd_path, point_cloud_from_mesh)
point_cloud = o3d.io.read_point_cloud(temp_pcd_path)

print(f"Number of vertices (points): {len(point_cloud.points)}")
print(f"Has color: {point_cloud.has_colors()}")

print("\nDisplaying point cloud...")
o3d.visualization.draw_geometries([point_cloud], window_name="Step 2: Point Cloud")

print("\nUnderstanding: Model converted to point cloud. Point cloud is a set of points in 3D space that describe the shape of the object without information about connections between points.\n")

print("STEP 3: Surface Reconstruction from Point Cloud")
print("-" * 80)

print("Performing surface reconstruction using Poisson method...")
poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    point_cloud, depth=9
)

print("Removing artifacts...")
vertices_to_remove = densities < np.quantile(densities, 0.01)
poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

bbox = point_cloud.get_axis_aligned_bounding_box()
poisson_mesh = poisson_mesh.crop(bbox)

poisson_mesh.compute_vertex_normals()

vertices = np.asarray(poisson_mesh.vertices)
y_min = np.min(vertices[:, 1])
y_max = np.max(vertices[:, 1])
y_range = y_max - y_min
colors = np.zeros((len(vertices), 3))
for i, vertex in enumerate(vertices):
    y = vertex[1]
    normalized_y = (y - y_min) / y_range
    colors[i] = [0.0, normalized_y, 1.0 - normalized_y]
poisson_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

print(f"Number of vertices: {len(poisson_mesh.vertices)}")
print(f"Number of triangles: {len(poisson_mesh.triangles)}")
print(f"Has color: {poisson_mesh.has_vertex_colors()}")

print("\nDisplaying reconstructed mesh...")
o3d.visualization.draw_geometries([poisson_mesh], window_name="Step 3: Reconstructed Mesh")

print("\nUnderstanding: Mesh reconstructed from point cloud using Poisson method. This method creates a smooth surface by connecting points with triangles. Artifacts (vertices with low density) were removed to improve model quality.\n")

print("STEP 4: Voxelization")
print("-" * 80)

bbox = point_cloud.get_axis_aligned_bounding_box()
diagonal = bbox.get_extent()
max_extent = np.max(diagonal)
voxel_size = max_extent / 100.0
print(f"Model size (max extent): {max_extent:.4f}")
print(f"Voxel size: {voxel_size:.4f}")

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)

voxels = voxel_grid.get_voxels()
print(f"Number of voxels: {len(voxels)}")
print(f"Has color: {voxel_grid.has_colors()}")

voxel_mesh = o3d.geometry.TriangleMesh()
voxels_list = voxel_grid.get_voxels()

bbox = point_cloud.get_axis_aligned_bounding_box()
origin = bbox.get_min_bound()

print(f"Creating colored mesh from {len(voxels_list)} voxels...")
for i, voxel in enumerate(voxels_list):
    cube = o3d.geometry.TriangleMesh.create_box(
        width=voxel_size, height=voxel_size, depth=voxel_size
    )
    voxel_coord = voxel.grid_index
    world_coord = origin + (np.array(voxel_coord) + 0.5) * voxel_size
    cube.translate(world_coord - np.array([voxel_size/2, voxel_size/2, voxel_size/2]))
    voxel_mesh += cube
    
    if (i + 1) % 5000 == 0:
        print(f"  Processed {i + 1}/{len(voxels_list)} voxels...")

voxel_mesh = voxel_mesh.remove_duplicated_vertices()
voxel_mesh = voxel_mesh.remove_duplicated_triangles()
voxel_mesh.compute_vertex_normals()

voxel_vertices = np.asarray(voxel_mesh.vertices)
x_min = np.min(voxel_vertices[:, 0])
x_max = np.max(voxel_vertices[:, 0])
x_range = x_max - x_min
voxel_colors = np.zeros((len(voxel_vertices), 3))
for i, vertex in enumerate(voxel_vertices):
    x = vertex[0]
    normalized_x = (x - x_min) / x_range
    voxel_colors[i] = [normalized_x, 0.2, 0.9]
voxel_mesh.vertex_colors = o3d.utility.Vector3dVector(voxel_colors)

print("\nDisplaying voxel grid...")
o3d.visualization.draw_geometries([voxel_mesh], window_name="Step 4: Voxel Grid")

print("\nUnderstanding: Point cloud converted to voxel grid. Voxels are 3D cubes that represent the object in discrete form. Voxel size determines the detail level of representation.\n")

print("STEP 5: Adding a Plane")
print("-" * 80)

bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
min_bound = bbox.get_min_bound()
max_bound = bbox.get_max_bound()
extent = bbox.get_extent()

plane_thickness = 0.1
plane_height = extent[1] * 1.2
plane_depth = extent[2] * 1.2
plane_mesh = o3d.geometry.TriangleMesh.create_box(
    width=plane_thickness, 
    height=plane_height, 
    depth=plane_depth
)

plane_mesh.translate([
    center[0] - plane_thickness/2, 
    min_bound[1] - extent[1] * 0.1, 
    center[2] - plane_depth/2
])
plane_mesh.compute_vertex_normals()
plane_mesh.paint_uniform_color([0.5, 0.5, 0.5])

mesh_vertices = np.asarray(mesh.vertices)
z_min = np.min(mesh_vertices[:, 2])
z_max = np.max(mesh_vertices[:, 2])
z_range = z_max - z_min
mesh_colors = np.zeros((len(mesh_vertices), 3))
for i, vertex in enumerate(mesh_vertices):
    z = vertex[2]
    normalized_z = (z - z_min) / z_range
    mesh_colors[i] = [normalized_z * 0.5, normalized_z * 0.5, 0.7]
mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

print(f"Plane created and placed between buildings")
print(f"Plane position (X center): {center[0]:.2f}")
print(f"Plane size: {plane_thickness:.2f} x {plane_height:.2f} x {plane_depth:.2f}")

print("\nDisplaying model with plane...")
o3d.visualization.draw_geometries(
    [mesh, plane_mesh],
    window_name="Step 5: Model with Plane"
)

print("\nUnderstanding: Vertical plane (box) created and placed between buildings. The plane will be used for clipping the model in the next step.\n")

print("STEP 6: Surface Clipping")
print("-" * 80)

bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
max_bound = bbox.get_max_bound()
min_bound = bbox.get_min_bound()

clip_bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=[min_bound[0], min_bound[1], min_bound[2]],
    max_bound=[center[0], max_bound[1], max_bound[2]]
)

clipped_mesh = mesh.crop(clip_bbox)

clipped_mesh.compute_vertex_normals()

clipped_vertices = np.asarray(clipped_mesh.vertices)
x_min = np.min(clipped_vertices[:, 0])
x_max = np.max(clipped_vertices[:, 0])
x_range = x_max - x_min
clipped_colors = np.zeros((len(clipped_vertices), 3))
for i, vertex in enumerate(clipped_vertices):
    x = vertex[0]
    normalized_x = (x - x_min) / x_range
    clipped_colors[i] = [1.0, normalized_x * 0.5, 0.0]
clipped_mesh.vertex_colors = o3d.utility.Vector3dVector(clipped_colors)

print(f"Number of remaining vertices: {len(clipped_mesh.vertices)}")
print(f"Number of triangles: {len(clipped_mesh.triangles)}")
print(f"Has color: {clipped_mesh.has_vertex_colors()}")
print(f"Has normals: {clipped_mesh.has_vertex_normals()}")

print("\nDisplaying clipped model...")
o3d.visualization.draw_geometries([clipped_mesh], window_name="Step 6: Clipped Model")

print("\nUnderstanding: Model clipped by vertical plane. All points lying to the right of the plane (in the positive X direction) were removed. This demonstrates the clipping operation in 3D space.\n")

print("STEP 7: Working with Color and Extremes")
print("-" * 80)

work_mesh = mesh

bbox = work_mesh.get_axis_aligned_bounding_box()
extent = bbox.get_extent()

vertices = np.asarray(work_mesh.vertices)

z_min = np.min(vertices[:, 2])
z_max = np.max(vertices[:, 2])
z_range = z_max - z_min

print(f"Minimum Z coordinate: {z_min:.4f}")
print(f"Maximum Z coordinate: {z_max:.4f}")

colors = np.zeros((len(vertices), 3))
for i, vertex in enumerate(vertices):
    z = vertex[2]
    normalized_z = (z - z_min) / z_range
    colors[i] = [normalized_z, 0.0, 1.0 - normalized_z]

work_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

min_z_idx = np.argmin(vertices[:, 2])
max_z_idx = np.argmax(vertices[:, 2])

min_point = vertices[min_z_idx]
max_point = vertices[max_z_idx]

print(f"\nExtreme points along Z axis:")
print(f"Minimum: ({min_point[0]:.4f}, {min_point[1]:.4f}, {min_point[2]:.4f})")
print(f"Maximum: ({max_point[0]:.4f}, {max_point[1]:.4f}, {max_point[2]:.4f})")

cube_size = np.max(extent) * 0.02
min_cube = o3d.geometry.TriangleMesh.create_box(
    width=cube_size, height=cube_size, depth=cube_size
)
min_cube.translate(min_point - np.array([cube_size/2, cube_size/2, cube_size/2]))
min_cube.paint_uniform_color([0.0, 1.0, 0.0])
min_cube.compute_vertex_normals()

max_cube = o3d.geometry.TriangleMesh.create_box(
    width=cube_size, height=cube_size, depth=cube_size
)
max_cube.translate(max_point - np.array([cube_size/2, cube_size/2, cube_size/2]))
max_cube.paint_uniform_color([1.0, 1.0, 0.0])
max_cube.compute_vertex_normals()

print("\nDisplaying model with gradient and extreme points...")
o3d.visualization.draw_geometries(
    [work_mesh, min_cube, max_cube],
    window_name="Step 7: Gradient and Extremes"
)

print("\nUnderstanding: New gradient applied along Z axis (from blue to red), replacing original colors. Extreme points found and highlighted: minimum (green cube) and maximum (yellow cube) along Z axis.\n")

print("=" * 80)
print("All steps completed successfully!")
print("=" * 80)