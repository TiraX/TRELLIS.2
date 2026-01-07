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
import glob

# # 1. Setup Environment Map
# envmap = EnvMap(torch.tensor(
#     cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
#     dtype=torch.float32, device='cuda'
# ))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

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

def image_to_mesh(image_path: str, output_path: str, verbose: bool = True):
    """
    Convert an image to a 3D mesh and export as GLB file.
    
    Args:
        image_path: Path to input image file (webp or png)
        output_path: Path to output GLB file
        verbose: Whether to print verbose messages
    """
    if os.path.exists(output_path):
        print(f"Skipping file already exists: {output_path}")
        return
    
    if verbose:
        print(f"\nProcessing: {image_path}")
    
    # Create an empty temporary GLB file
    with open(output_path, 'w') as f:
        pass
    
    if verbose:
        print(f"Created temporary file: {output_path}")
    
    try:
        # Load Image & Run
        image = Image.open(image_path)
        mesh = pipeline.run(image=image, tex_slat_sampler_params = {"steps": 1}, pipeline_type='1024_cascade')[0]
        mesh.simplify(16777216)  # nvdiffrast limit
        
        # Export to GLB
        glb = to_glb_cpu_vertex_only(
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
            verbose             =   verbose
        )
        
        # Delete the temporary file before saving the real GLB
        if os.path.exists(output_path):
            os.remove(output_path)
            if verbose:
                print(f"Deleted temporary file: {output_path}")
        
        glb.export(output_path, extension_webp=False)
        
        if verbose:
            print(f"Saved to: {output_path}")
    except Exception as e:
        # Clean up temporary file if an error occurs
        if os.path.exists(output_path):
            os.remove(output_path)
            if verbose:
                print(f"Cleaned up temporary file due to error: {output_path}")
        raise e

def main():
    """
    Process all webp and png images in assets/example_image directory.
    """
    input_dir = "assets/example_image"
    output_dir = "output_glb"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all webp and png files
    image_files = []
    for ext in ['*.webp', '*.png']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"No webp or png files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    
    # Process each image
    for image_path in image_files:
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.glb")
        
        try:
            image_to_mesh(image_path, output_path, verbose=True)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"\nAll done! Processed {len(image_files)} image(s)")

if __name__ == "__main__":
    main()