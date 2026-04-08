import open3d as o3d
import trimesh
import json
import os
import numpy as np
from plyfile import PlyData
import heapq
# import bpy
# from mathutils import Vector
import os
import pyvista as pv
from vtkmodules.vtkRenderingFreeType import vtkVectorText

from vtkmodules.vtkFiltersModeling import vtkLinearExtrusionFilter
import random
import copy
import colorsys
import cv2
import glob
import shutil
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

matport_names = ['17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '2azQ1b91cZZ', '2n8kARJN3HM', '2t7WUuJeko7', '5LpN3gDmAk7', 
                     '5q7pvUzZiYa', '5ZKStnWn8Zo', '759xd9YjKW5', '7y3sRwLe3Va', '8194nk5LbLH', '82sE5b5pLXE', '8WUmhLawc2A', 'aayBHfsNo7d', 
                      'ac26ZMwG7aT', 'ARNzJeq3xxb', 'B6ByNegPMKs', 'b8cTxDM8gDG', 'cV4RVeZvu5T', 'D7G3Y4RVNrH', 'D7N2EKCX4Sj', 'dhjEzFoUFzH', 
                      'E9uDoFAP3SH', 'e9zR4mvMWw7', 'EDJbREhghzL', 'EU6Fwq7SyZv', 'fzynW3qQPVF', 'GdvgFV5R1Z5', 'gTV8FGcVJC9', 'gxdoqLR6rwA', 
                      'gYvKGZ5eRqb', 'gZ6f7yhEvPG', 'HxpKQynjfin', 'i5noydFURQK', 'JeFG25nYj2p', 'JF19kD82Mey', 'jh4fc5c5qoQ', 'JmbYfDe2QKZ', 
                      'jtcxE69GiFV', 'kEZ7cmS4wCh', 'mJXqzFtmKg4', 'oLBMNvg9in8', 'p5wJjkQkbXX', 'pa4otMbVnkk', 'pLe4wQe7qrG', 'Pm6F8kyY3z2', 
                      'pRbA3pwrgk9', 'PuKPg4mmafe', 'PX4nDJXEHrG', 'q9vSo1VnCiC', 'qoiz87JEwZ2', 'QUCTc6BB5sX', 'r1Q1Z4BcV1o', 'r47D5H71a5s', 
                      'rPc6DW4iMge', 'RPmz2sHmrrY', 'rqfALeAoiTq', 's8pcmisQ38h', 'S9hNv5qa7GM', 'sKLMLpTHeUy', 'SN83YJsR3w2', 'sT4fr6TAbpF', 
                      'TbHJrupSAjP', 'ULsKaCPVFJR', 'uNb9QFRL6hY', 'ur6pFq6Qu1A', 'UwV83HsGsw3', 'Uxmj2M2itWa', 'V2XKFyX4ASd', 'VFuaQ6m2Qom', 
                      'VLzqgDo317F', 'Vt2qJdWjCF2', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'vyrNrziPKCB', 'VzqfbhrpDEA', 'wc2JMjhGNzB', 'WYY7iVyf5p8', 
                      'X7HyMhZNoso', 'x8F5xyUWy9e', 'XcA2TqTSSAj', 'YFuZgdQ5vWj', 'YmJkqBEsHnH', 'yqstnuAEVhm', 'YVUC4YcDtcY', 'Z6MFQCViBuw', 
                      'ZMojNkEp431', 'zsNo4HB9uLZ']
colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.4, 1.0], [1.0, 0.5, 0.0], [0.6, 0.2, 0.8], [0.0, 0.9, 0.9],
	[1.0, 0.0, 1.0], [0.9, 0.3, 0.1], [0.0, 0.6, 0.6], [0.5, 0.25, 0.05], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0],
	[0.5, 0.5, 0.0], [0.0, 0.0, 0.4], [0.0, 0.4, 0.0], [1.0, 0.4, 0.4], [0.27, 0.51, 0.71], [0.29, 0.0, 0.51],
 	[1.0, 0.08, 0.58], [0.85, 0.65, 0.13]])


def get_bbx_from_dict(obj):

    obj_name = obj['category']
    # print('obj_name:', obj_name)
    obj_path = obj['source']
    eightPoints = obj['obb']['eightPoints']
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(eightPoints))

    min_bound = aabb.get_min_bound() 
    max_bound = aabb.get_max_bound() 

    return min_bound, max_bound
   
def compute_bbox_volume(min_corner, max_corner):
    length = max_corner[0] - min_corner[0]
    width = max_corner[1] - min_corner[1]
    height = max_corner[2] - min_corner[2]
    volume = length * width * height
    return volume
def compute_bbox_area(min_corner, max_corner):
    length = max_corner[0] - min_corner[0]
    width = max_corner[1] - min_corner[1]
    height = max_corner[2] - min_corner[2]
    area1 = length * width
    area2 = length * height
    area3 = width * height
    return area1, area2, area3

def top_ranked_objects_by_bbox_volume_area(objs, rank_num=8):
    """Select the top objects ranked by bounding-box volume and area"""
    heap_volume = []
    heap_area = []
    obj_info = {}  # record (volume, area)

    for obj in objs:
        if obj['category'] in ['wall', 'ceiling', 'floor']:
            continue

        obj_name = obj['name']
        obj_uid = obj['uid']
        try:
            min_corner, max_corner = get_bbx_from_dict(obj)
            
            volume = compute_bbox_volume(min_corner, max_corner)
            area = max(compute_bbox_area(min_corner, max_corner))
        except Exception as e:
            print(f"[Warning] Failed to compute bbox for {obj.name}: {e}")
            continue

        obj_info[obj_name] = (volume, area)


        if len(heap_volume) < rank_num:
          
            heapq.heappush(heap_volume, (volume, obj_uid, obj_name))
        else:
         
            heapq.heappushpop(heap_volume, (volume, obj_uid, obj_name))

        if len(heap_area) < rank_num:
          
            heapq.heappush(heap_area, (area, obj_uid, obj_name))
        else:
      
            heapq.heappushpop(heap_area, (area, obj_uid, obj_name))

    top_volume_objs = set(obj for _, __, obj in heap_volume)
    top_area_objs = set(obj for _, __, obj in heap_area)                            

    combined_objs = top_volume_objs.union(top_area_objs)

    result = []
    for obj in combined_objs:
        volume, area = obj_info[obj]
        result.append([obj, volume, area])

    result.sort(key=lambda x: (-x[1], -x[2]))
    
    return result[:rank_num]


def create_vtk_text(text, position, max_z, bbx_z, which_view, control=0.25, scale=0.4, color=[1,0,0]):  
    vec_text = vtkVectorText()
    vec_text.SetText(text)
    vec_text.Update()

    extrude = vtkLinearExtrusionFilter()
    extrude.SetInputConnection(vec_text.GetOutputPort())
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0, 0, 1)  # Z 
    extrude.SetScaleFactor(0.15)  
    extrude.SetCapping(True) 
    extrude.Update()

    label_mesh = pv.wrap(extrude.GetOutput())
    label_mesh.scale([scale, scale, scale], inplace=True)
    label_mesh.translate(position, inplace=True)

    label_mesh_copy = label_mesh.copy() 
    label_mesh1 = label_mesh_copy

    tmp_center = label_mesh_copy.center
    if which_view == 'front':
        angle = 0
        angle_lean =0 #60
    elif which_view == 'back':
        angle = -180
        angle_lean = -60
    elif which_view == 'left':  # -90
        angle = -90
        angle_lean =-45
    elif which_view == 'right':
        angle = -270
        angle_lean =45
    elif which_view == 'all':  
        angle = 0
        angle_lean =0
    elif which_view == 'top-down':  
        angle = 0
        angle_lean =0

    if which_view in ['front', 'back']:
        label_mesh1.rotate_z(angle, point=tmp_center, inplace=True)
        label_mesh_copy.rotate_x(angle_lean, point=tmp_center, inplace=True)
    elif which_view in ['left', 'right']:
        label_mesh1.rotate_z(angle, point=tmp_center, inplace=True)
        label_mesh_copy.rotate_y(angle_lean, point=tmp_center, inplace=True)
    
    # rgb = (np.array((1.0, 0.0, 0.0)) * 255).astype(np.uint8)
    rgb = (color * 255).astype(np.uint8)
    label_mesh1.point_data["RGB"] = np.tile(rgb, (label_mesh.n_points, 1))
    label_mesh1.active_scalars_name = "RGB"


    return label_mesh1


def pv_to_o3d(pv_mesh):
    pv_mesh = pv_mesh.triangulate()
    verts = np.asarray(pv_mesh.points)
    faces = np.asarray(pv_mesh.faces.reshape(-1, 4))[:, 1:]  # triangle faces
    colors = np.asarray(pv_mesh.point_data['RGB']) / 255.0

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return o3d_mesh


def screenshot(mesh, center_text=None, which_view='top-down', save_dir="./output_views", panorama_name='test'):
    
    width, height = 3840, 2160  #1024, 1024   # 3840, 2160  #
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit' 

    material.point_size = 1.0  # 3
    scene.add_geometry("mesh", mesh, material)

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()

    diag = np.linalg.norm(extent)


    cam_dir = np.array([0, 0, 1])  
    
    fov_rad = np.deg2rad(60.0)  
    scale_factor = 0.5 / np.tan(fov_rad / 2.0) 

    cam_pos = center + cam_dir * diag * scale_factor


    up = [1, 0, 0]   #[0, 1, 0]  # Y  up
    scene.camera.look_at(center, cam_pos, up)  

    proj_proj = renderer.scene.camera.get_projection_matrix()
    
    view_view = renderer.scene.camera.get_view_matrix()
    order_center = np.array(center_text[0])  
    order_color = np.array(center_text[1])  
    order_category = np.array(center_text[2])  

    pixel_x, pixel_y = project_points(order_center, view_view, proj_proj, width, height)
    points = [(int(xi), int(yi)) for xi, yi in zip(pixel_x, pixel_y)]
  
    image = renderer.render_to_image()
    image = draw_labels_on_image(image, points, order_color, order_category)

    path = os.path.join(save_dir, f'{panorama_name}_view_{which_view}.png')
  
    cv2.imwrite(path, image)

    crop_image(path) 

def project_points(points_3d, view, proj, width, height):
       
        ones = np.ones((points_3d.shape[0], 1))
        points_h = np.hstack([points_3d, ones])                      # shape (8, 4)

        points_cam = (view @ points_h.T).T                           # shape (8, 4)

        points_clip = (proj @ points_cam.T).T                        # shape (8, 4)

        points_ndc = points_clip[:, :3] / points_clip[:, 3][:, None] # shape (8, 3)

        u = (points_ndc[:, 0] + 1) * 0.5 * width
        v = (1 - points_ndc[:, 1]) * 0.5 * height

        # return np.stack([u, v], axis=1)  
        return u, v

def compute_cam_pos(center, direction, bbox_extent, fov_deg=60.0, margin_ratio=1.2):
    diag_len = np.linalg.norm(bbox_extent)
    fov_rad = np.deg2rad(fov_deg)
    distance = (diag_len * margin_ratio) / (2 * np.tan(fov_rad / 2))

    cam_pos = center + direction / np.linalg.norm(direction) * distance
    return cam_pos

def render_four_oblique_views(mesh, center_text=None, which_view='all', save_dir="./output_views", panorama_name='test'):
    width, height = 3840, 2160
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'
    material.point_size = 1.0
    scene.add_geometry("mesh", mesh, material)

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()

    directions_dict = {
        "front": np.array([0, -1, 1]),
        "back":  np.array([0,  1, 1]),
        "left":  np.array([-1, 0, 1]),
        "right": np.array([ 1, 0, 1]),
        "top-down": np.array([0, 0, 1])
    }

    if which_view == 'all':
        views = ["front", "back", "left", "right"]
    elif which_view in directions_dict:
        views = [which_view]
    else:
        raise ValueError(f"Unknown view: {which_view}")

    os.makedirs(save_dir, exist_ok=True)

    for name in views:
        direction = directions_dict[name]
        cam_pos = compute_cam_pos(center, direction, extent)
        up = [0, 0, 1]

        scene.camera.look_at(center, cam_pos, up)

        
        proj_proj = renderer.scene.camera.get_projection_matrix()
        view_view = renderer.scene.camera.get_view_matrix()

       
        order_center = np.array(center_text[0])  
        order_color = np.array(center_text[1])  
        order_category = np.array(center_text[2])  

        pixel_x, pixel_y = project_points(order_center, view_view, proj_proj, width, height)
        points = [(int(xi), int(yi)) for xi, yi in zip(pixel_x, pixel_y)]
        image = renderer.render_to_image()

        image = draw_labels_on_image(image, points, order_color, order_category)
        # cv2.imwrite("./no_overlap_labels.png", image)

        path = os.path.join(save_dir, f"{panorama_name}_view_{name}.png")
        cv2.imwrite(path, image)


        crop_image(path) 


def does_overlap(rect1, rect2):
    """Check whether two rectangles overlap."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def is_in_bounds(rect, img_width, img_height):
    """Check whether it is within the image bounds."""
    x, y, w, h = rect
    return x >= 0 and y >= 0 and (x + w) <= img_width and (y + h) <= img_height

def find_valid_position(base_pos, text_size, existing_boxes, img_shape, max_radius=100):
    """Find a non-overlapping position with minimal displacement."""
    w, h = text_size
    img_h, img_w = img_shape[:2]
    cx, cy = base_pos

    for radius in range(0, max_radius + 1, 2):
        for dx in range(-radius, radius + 1, 2):
            for dy in range(-radius, radius + 1, 2):
                x = cx + dx
                y = cy + dy
                rect = (x, y - h, w, h)  
                if is_in_bounds(rect, img_w, img_h) and all(not does_overlap(rect, b) for b in existing_boxes):
                    return (x, y), rect
    return None, None

def draw_labels_on_image(img, positions, order_color, labels, font_scale=1.0, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    img = np.array(img)
    existing_boxes = []
    for pos, color, label in zip(positions, order_color, labels):
        size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        draw_pos, rect = find_valid_position(pos, size, existing_boxes, img.shape)
        if draw_pos:
            # cv2.putText(img, label, draw_pos, font, font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)
            cv2.putText(img, label, draw_pos, font, font_scale, color*255, thickness, lineType=cv2.LINE_AA)
            existing_boxes.append(rect)
        else:
            print(f"Label [{label}] could not be placed in the image (overlapping and out of bounds).")

    return img


def crop_image(path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)  #200

    coords = cv2.findNonZero(thresh)  #
    x, y, w, h = cv2.boundingRect(coords)  #

    offset = 50
    
    cropped = img[y-offset:y+h+offset, x-offset:x+w+offset]

    try:
        cv2.imwrite(path, cropped)
    except:
        pass


def create_line_cylinder(p1, p2, radius=0.01, resolution=8):
    """Simulate a line segment from p1 to p2 using a cylinder, without relying on high-version Open3D functions."""
    
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0:
        return o3d.geometry.TriangleMesh()

    direction = vec / length


    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
    cylinder.compute_vertex_normals()

  
    z = np.array([0, 0, 1])
    v = np.cross(z, direction)
    c = np.dot(z, direction)
    s = np.linalg.norm(v)

    if s == 0:
        R = np.eye(3) if c > 0 else -np.eye(3)  
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    cylinder.rotate(R, center=np.array([0, 0, 0]))

    mid = (p1 + p2) / 2
    cylinder.translate(mid)

    return cylinder


def create_bbx_lines_as_mesh(points, colors, save_path=None):
    # Create a 3D bounding box mesh from corner points using cylinder edges and optionally save it
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # sides
    ]

    all_meshes = []

    for start_idx, end_idx in lines:
        p1 = np.array(points[start_idx])
        p2 = np.array(points[end_idx])
        cyl = create_line_cylinder(p1, p2, radius=0.01)  #0.005
        cyl.paint_uniform_color(colors)  # black
        all_meshes.append(cyl)

    final_mesh = all_meshes[0]
    for mesh in all_meshes[1:]:
        final_mesh += mesh

    if save_path:
        o3d.io.write_triangle_mesh(save_path, final_mesh)

    return final_mesh


def create_rotated_bbox(centroid, size, rotation, colors):
    # draw 3D bbox
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = centroid
    bbox.extent = size
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rotation])
    bbox.R = R

    bbox.color = colors    #[0, 0, 1] 
    return bbox

def main(root_path, save_path): 
    os.makedirs(save_path, exist_ok=True)
    
    control = 0.25  # 控制字体的z轴

    with open(root_path, 'r', encoding='utf-8') as f:   
        data = json.load(f)

    len_cur_obj_num = {}  # Check the number of objects in each region to avoid cases with only walls/ceilings or too few objects
   
    for region in data['scene']: 

        region_name = region['category']
        print('region_name:', region_name) 
        region_path = region['source']
        region_id = os.path.basename(os.path.dirname(region_path))

        if '/' in region_name:
            region_name = region_name.replace('/', '-')
        
        panorama_name = f"{region_name}_{region_id}"


        objs = region['bbox']
        result = top_ranked_objects_by_bbox_volume_area(objs, rank_num=1000)  # Select the top objects ranked by bounding-box volume and area, rank_num:optional
        
        sort_name = [row[0] for row in result]
        len_cur_obj_num[panorama_name] = len(sort_name) #

        max_z = 0
        # five render view: "front", "back", "left", "right", 'top-down'
        mesh1 = o3d.geometry.TriangleMesh()
        mesh2 = o3d.geometry.TriangleMesh()
        mesh3 = o3d.geometry.TriangleMesh()
        mesh4 = o3d.geometry.TriangleMesh()
        mesh5 = o3d.geometry.TriangleMesh()

        

        order_center = []
        order_color = []
        order_category = []


        tmp_test = []
        for i, obj in enumerate(objs):
            if obj['name'] in sort_name: 
                eightpoint = obj['obb']['eightPoints']

                centroid_new = obj['obb']['centroid']  
                size_new = obj['obb']['axisSize']   # axesLengths     axisSize
                rotation = obj['obb']['rotation']  

                

                idx = i % 20  #20 colors
                color = colors[idx]
                order_color.append(color)
                boxs = create_bbx_lines_as_mesh(eightpoint, color)

                boxs1 = create_rotated_bbox(centroid_new, size_new, rotation, color)  # draw bbx
                tmp_test.append(boxs1)

                mesh1 += boxs
                mesh2 += boxs
                mesh3 += boxs
                mesh4 += boxs
                mesh5 += boxs

                # all_bbx.append(boxs)
              
                category = obj['category']
                order_category.append(category)
                # print((l2_norm, category))
                text_point = obj['obb']['centroid']
                order_center.append(text_point)
                bbx_z = obj['obb']['axesLengths'][2]
                text_dict = {}
                for view in ["front", "back", "left", "right", 'top-down']:
                    label_mesh = create_vtk_text(category, text_point, max_z, bbx_z, view, control, scale=0.15, color=color)  #创建label字体
                    text_o3d = pv_to_o3d(label_mesh) # 转mesh为o3d
                    # o3d.visualization.draw_geometries([text_o3d])

                    text_dict[view] = text_o3d
               
                tmp_test.append(text_dict["front"])
       
        o3d.visualization.draw_geometries(tmp_test)
       
    
        # render_four_oblique_views(mesh1, center_text=[order_center, order_color, order_category], which_view='front', save_dir=save_path, panorama_name=panorama_name) 

     
        # render_four_oblique_views(mesh2, center_text=[order_center, order_color, order_category], which_view='back', save_dir=save_path, panorama_name=panorama_name)

      
        # render_four_oblique_views(mesh3, center_text=[order_center, order_color, order_category], which_view='left', save_dir=save_path, panorama_name=panorama_name)

        # render_four_oblique_views(mesh4, center_text=[order_center, order_color, order_category], which_view='right', save_dir=save_path, panorama_name=panorama_name)

    
        screenshot(mesh5, center_text=[order_center, order_color, order_category], which_view='top-down', save_dir=save_path, panorama_name=panorama_name)
     
    return len_cur_obj_num

            
    

if __name__ == "__main__":
    matport_names = ['17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '2azQ1b91cZZ', '2n8kARJN3HM', '2t7WUuJeko7', '5LpN3gDmAk7', 
                     '5q7pvUzZiYa', '5ZKStnWn8Zo', '759xd9YjKW5', '7y3sRwLe3Va', '8194nk5LbLH', '82sE5b5pLXE', '8WUmhLawc2A', 'aayBHfsNo7d', 
                      'ac26ZMwG7aT', 'ARNzJeq3xxb', 'B6ByNegPMKs', 'b8cTxDM8gDG', 'cV4RVeZvu5T', 'D7G3Y4RVNrH', 'D7N2EKCX4Sj', 'dhjEzFoUFzH', 
                      'E9uDoFAP3SH', 'e9zR4mvMWw7', 'EDJbREhghzL', 'EU6Fwq7SyZv', 'fzynW3qQPVF', 'GdvgFV5R1Z5', 'gTV8FGcVJC9', 'gxdoqLR6rwA', 
                      'gYvKGZ5eRqb', 'gZ6f7yhEvPG', 'HxpKQynjfin', 'i5noydFURQK', 'JeFG25nYj2p', 'JF19kD82Mey', 'jh4fc5c5qoQ', 'JmbYfDe2QKZ', 
                      'jtcxE69GiFV', 'kEZ7cmS4wCh', 'mJXqzFtmKg4', 'oLBMNvg9in8', 'p5wJjkQkbXX', 'pa4otMbVnkk', 'pLe4wQe7qrG', 'Pm6F8kyY3z2', 
                      'pRbA3pwrgk9', 'PuKPg4mmafe', 'PX4nDJXEHrG', 'q9vSo1VnCiC', 'qoiz87JEwZ2', 'QUCTc6BB5sX', 'r1Q1Z4BcV1o', 'r47D5H71a5s', 
                      'rPc6DW4iMge', 'RPmz2sHmrrY', 'rqfALeAoiTq', 's8pcmisQ38h', 'S9hNv5qa7GM', 'sKLMLpTHeUy', 'SN83YJsR3w2', 'sT4fr6TAbpF', 
                      'TbHJrupSAjP', 'ULsKaCPVFJR', 'uNb9QFRL6hY', 'ur6pFq6Qu1A', 'UwV83HsGsw3', 'Uxmj2M2itWa', 'V2XKFyX4ASd', 'VFuaQ6m2Qom', 
                      'VLzqgDo317F', 'Vt2qJdWjCF2', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'vyrNrziPKCB', 'VzqfbhrpDEA', 'wc2JMjhGNzB', 'WYY7iVyf5p8', 
                      'X7HyMhZNoso', 'x8F5xyUWy9e', 'XcA2TqTSSAj', 'YFuZgdQ5vWj', 'YmJkqBEsHnH', 'yqstnuAEVhm', 'YVUC4YcDtcY', 'Z6MFQCViBuw', 
                      'ZMojNkEp431', 'zsNo4HB9uLZ']


    root_path = './data_matterport3d/17DRP5sb8fy/building_level.json'  # Processed building/region/room information
    save_path = './3Dbbox_top_down_image/17DRP5sb8fy'   # save path of rendered image 

    replace = '17DRP5sb8fy'
    i = 0
    out_num = []
    for build_name in tqdm(matport_names):
        i = i+1
        print((i, build_name))
        root_path = root_path.replace(replace, build_name)
        save_path = save_path.replace(replace, build_name)
        
        replace = build_name
        len_cur_obj_num = main(root_path, save_path)
        out_num.append({build_name:len_cur_obj_num})
    


