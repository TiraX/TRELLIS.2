import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
from tqdm import tqdm
import trimesh
from typing import *
import numpy as np

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Image & Run
image = Image.open("assets/example_image/a306e2ee5cbc3da45e7db48d75a0cade0bb7eee263a74bc6820c617afaba1302.webp")
mesh = pipeline.run(image=image, pipeline_type='1024_cascade')[0]
mesh.simplify(16777216) # nvdiffrast limit

# # 4. Render Video
# video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
# imageio.mimsave("sample.mp4", video, fps=15)

def to_glb_cpu(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file using CPU processing.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sparse tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing (currently not supported on CPU)
        remesh_band: size of the remeshing band (currently not supported on CPU)
        remesh_project: projection factor for remeshing (currently not supported on CPU)
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    try:
        import xatlas
    except ImportError:
        raise ImportError("xatlas is required for CPU-based UV unwrapping. Install it with: pip install xatlas")
    
    from scipy.spatial import cKDTree
    
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB (CPU)")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to CPU
    vertices = vertices.cpu()
    faces = faces.cpu()
    attr_volume = attr_volume.cpu()
    coords = coords.cpu()
    aabb = aabb.cpu()
    voxel_size = voxel_size.cpu()
    grid_size = grid_size.cpu()
    
    # Convert to numpy arrays
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    
    # --- Initial Mesh Cleaning ---
    if verbose:
        print("Cleaning mesh...")
    
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    
    # Fill holes
    mesh.fill_holes()
    if verbose:
        print(f"After filling holes: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    if use_tqdm:
        pbar.update(1)
    
    # --- Mesh Simplification ---
    if use_tqdm:
        pbar.set_description("Simplifying mesh")
    if verbose:
        print("Simplifying mesh...")
    
    if not remesh:
        # Aggressive simplification (3x target)
        if len(mesh.faces) > decimation_target * 3:
            target_ratio = (decimation_target * 3) / len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(target_ratio)
            if verbose:
                print(f"After initial simplification: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Fill holes again after cleanup
        mesh.fill_holes()
        if verbose:
            print(f"After initial cleanup: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Final simplification to target count
        if len(mesh.faces) > decimation_target:
            target_ratio = decimation_target / len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(target_ratio)
            if verbose:
                print(f"After final simplification: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Final cleanup
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fill_holes()
        if verbose:
            print(f"After final cleanup: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Fix normals
        mesh.fix_normals()
    else:
        if verbose:
            print("Warning: Remeshing is not supported in CPU mode. Skipping remeshing.")
        # Just simplify
        if len(mesh.faces) > decimation_target:
            target_ratio = decimation_target / len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(target_ratio)
            if verbose:
                print(f"After simplification: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing mesh")
    if verbose:
        print("Parameterizing mesh...")
    
    # Use xatlas for UV unwrapping
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    
    out_vertices = mesh.vertices[vmapping]
    out_faces = indices
    out_uvs = uvs
    
    # Compute normals
    mesh_with_uv = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
    mesh_with_uv.vertex_normals  # This computes normals
    out_normals = mesh_with_uv.vertex_normals
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Build KDTree for nearest neighbor search ---
    if use_tqdm:
        pbar.set_description("Building KDTree")
    if verbose:
        print("Building KDTree for original mesh...", end='', flush=True)
    
    # Build KDTree on original vertices
    orig_vertices_np = vertices.numpy()
    orig_faces_np = faces.numpy()
    kdtree = cKDTree(orig_vertices_np)
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)
    
    # Create a simple rasterizer using numpy
    # For each pixel in the texture, we need to find which triangle it belongs to
    # and interpolate the 3D position
    
    texture_h, texture_w = texture_size, texture_size
    attrs = np.zeros((texture_h, texture_w, attr_volume.shape[1]), dtype=np.float32)
    mask = np.zeros((texture_h, texture_w), dtype=bool)
    
    # Rasterize triangles in UV space
    for face_idx in range(len(out_faces)):
        face = out_faces[face_idx]
        uv_tri = out_uvs[face]  # (3, 2)
        pos_tri = out_vertices[face]  # (3, 3)
        
        # Convert UV to pixel coordinates
        uv_pixels = uv_tri * np.array([texture_w - 1, texture_h - 1])
        
        # Get bounding box
        min_x = int(np.floor(uv_pixels[:, 0].min()))
        max_x = int(np.ceil(uv_pixels[:, 0].max()))
        min_y = int(np.floor(uv_pixels[:, 1].min()))
        max_y = int(np.ceil(uv_pixels[:, 1].max()))
        
        # Clamp to texture bounds
        min_x = max(0, min_x)
        max_x = min(texture_w - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(texture_h - 1, max_y)
        
        # Rasterize pixels in bounding box
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Check if pixel center is inside triangle
                px = x + 0.5
                py = y + 0.5
                
                # Compute barycentric coordinates
                v0 = uv_pixels[1] - uv_pixels[0]
                v1 = uv_pixels[2] - uv_pixels[0]
                v2 = np.array([px, py]) - uv_pixels[0]
                
                d00 = np.dot(v0, v0)
                d01 = np.dot(v0, v1)
                d11 = np.dot(v1, v1)
                d20 = np.dot(v2, v0)
                d21 = np.dot(v2, v1)
                
                denom = d00 * d11 - d01 * d01
                if abs(denom) < 1e-10:
                    continue
                
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1.0 - v - w
                
                # Check if inside triangle
                if u >= -1e-6 and v >= -1e-6 and w >= -1e-6:
                    # Interpolate 3D position
                    pos_3d = u * pos_tri[0] + v * pos_tri[1] + w * pos_tri[2]
                    
                    # Find nearest point on original mesh
                    _, nearest_idx = kdtree.query(pos_3d)
                    nearest_vert = orig_vertices_np[nearest_idx]
                    
                    # Sample from attribute volume using trilinear interpolation
                    # Convert position to voxel coordinates
                    voxel_coord = (nearest_vert - aabb[0].numpy()) / voxel_size.numpy()
                    
                    # Clamp to grid bounds
                    voxel_coord = np.clip(voxel_coord, 0, grid_size.numpy() - 1)
                    
                    # Find nearest voxel in sparse coords
                    coords_np = coords.numpy()
                    # Simple nearest neighbor in voxel space
                    voxel_coord_int = np.round(voxel_coord).astype(int)
                    
                    # Find matching coordinate in sparse tensor
                    matches = np.all(coords_np == voxel_coord_int, axis=1)
                    if np.any(matches):
                        attr_idx = np.where(matches)[0][0]
                        attrs[y, x] = attr_volume[attr_idx].numpy()
                        mask[y, x] = True
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    # Extract channels based on layout
    base_color = np.clip(attrs[..., attr_layout['base_color']] * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']] * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']] * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']] * 255, 0, 255).astype(np.uint8)
    alpha_mode = 'OPAQUE'
    
    # Inpainting: fill gaps to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    
    # Create PBR material
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
    
    # --- Coordinate System Conversion & Final Object ---
    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    out_vertices[:, 1], out_vertices[:, 2] = out_vertices[:, 2].copy(), -out_vertices[:, 1].copy()
    out_normals[:, 1], out_normals[:, 2] = out_normals[:, 2].copy(), -out_normals[:, 1].copy()
    out_uvs[:, 1] = 1 - out_uvs[:, 1]  # Flip UV V-coordinate
    
    textured_mesh = trimesh.Trimesh(
        vertices=out_vertices,
        faces=out_faces,
        vertex_normals=out_normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=out_uvs, material=material)
    )
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")
    
    return textured_mesh

def to_glb_cpu_vertex_only(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file using CPU processing.
    This is a simplified version that only saves vertices and faces as a basic mesh.
    
    Note: This function currently ignores remesh, fill_holes, simplify, and material sampling.
    These features will be added in future updates.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sparse tensor for attribute interpolation (currently unused)
        coords: (L, 3) tensor of coordinates for each voxel (currently unused)
        attr_layout: dictionary of slice objects for each attribute (currently unused)
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume (currently unused)
        voxel_size: (3,) tensor of size of each voxel (currently unused)
        grid_size: (3,) tensor of number of voxels in each dimension (currently unused)
        decimation_target: target number of vertices for mesh simplification (currently unused)
        texture_size: size of the texture for baking (currently unused)
        remesh: whether to perform remeshing (currently unused)
        remesh_band: size of the remeshing band (currently unused)
        remesh_project: projection factor for remeshing (currently unused)
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering (currently unused)
        mesh_cluster_refine_iterations: number of iterations for refining clusters (currently unused)
        mesh_cluster_global_iterations: number of global iterations for clustering (currently unused)
        mesh_cluster_smooth_strength: strength of smoothing for clustering (currently unused)
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
        
    Returns:
        trimesh.Trimesh: A basic mesh with vertices and faces
    """
    if use_tqdm:
        pbar = tqdm(total=2, desc="Creating GLB (CPU)")
    
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    
    # Move data to CPU if needed
    if vertices.is_cuda:
        vertices = vertices.cpu()
    if faces.is_cuda:
        faces = faces.cpu()
    
    if use_tqdm:
        pbar.update(1)
    
    # Convert to numpy arrays
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    
    if verbose:
        print("Converting coordinate system...")
    
    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    
    # Create a basic trimesh object
    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        process=False,  # Don't auto-process to preserve original geometry
    )
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    
    if verbose:
        print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print("Done")
    
    return mesh

# 5. Export to GLB
# glb = o_voxel.postprocess.to_glb(
# glb = to_glb_cpu_vertex_only(
glb = to_glb_cpu(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   2000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True,
    use_tqdm            =   True
)
glb.export("sample.glb", extension_webp=False)