import bpy
import os
from mathutils import Vector, geometry, Matrix
import bmesh
import math
import json
import numpy as np
import heapq

def get_object_center(obj):
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    center = sum(verts, Vector()) / len(verts)
    return center

def get_object_world_center(obj):
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    center = sum(verts, Vector()) / len(verts)
    return center

def normalize_object(obj):
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_corner = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    max_corner = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
    center = (min_corner + max_corner) / 2
    size = max((max_corner - min_corner).length, 1e-6)

    obj.location -= center
    obj.scale /= size
    bpy.context.view_layer.update()

def move_objects_to_origin(objects, move_vector):
    T = Matrix.Translation(-move_vector)
    
    for obj in objects:
        obj.matrix_world = T @ obj.matrix_world
    
    bpy.context.view_layer.update()

def normalize_for_gif(objs, target_size=5.0):
    x_max = y_max = z_max = -float("inf")
    x_min = y_min = z_min =  float("inf")

    for obj in objs:
        for v in obj.bound_box:
            world_co = obj.matrix_world @ Vector(v)
            x_max, y_max, z_max = max(x_max, world_co.x), max(y_max, world_co.y), max(z_max, world_co.z)
            x_min, y_min, z_min = min(x_min, world_co.x), min(y_min, world_co.y), min(z_min, world_co.z)

    max_dim = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_dim <= 1e-6:
        return                    

    scale_factor = target_size / max_dim

    S = Matrix.Scale(scale_factor, 4)      
    for obj in objs:
        obj.matrix_world = S @ obj.matrix_world

    bpy.context.view_layer.update()




def is_in_target_collection(obj, target_collection_name):
    for col in obj.users_collection:
        if col.name in target_collection_name:
            return True
        else:
            return False

# PIP: point in polygon: ray casting algorithm
def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def get_first_spawn_asset_location():
    for obj in bpy.data.objects:
        for col in obj.users_collection:
            if col.name == "unique_assets" and ".spawn_asset" in obj.name:
                print(f"Found spawn_asset object: {obj.name}")
                center = get_object_world_center(obj)
                # center = obj.location
                print(f"{obj}'s center is {center}")
                return center
    raise ValueError("No '.spawn_asset' object found in scene.")

def copy_objects_for_render():
    copied_objects = []
    for obj in bpy.data.objects:
        if not obj.hide_render:
            obj_copy = obj.copy()
            if obj.data:
                obj_copy.data = obj.data.copy()
            bpy.context.collection.objects.link(obj_copy)
            copied_objects.append(obj_copy)
    return copied_objects

def cleanup_copied_objects(objects):
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    print(f"[Cleanup] Deleted {len(objects)} copied objects.")

def render_scene():
    bpy.ops.render.render(write_still=True)
    print(f"[Render] Scene rendered to {bpy.context.scene.render.filepath}") 

def generate_semisphere_camera(x, y, z_height, angle):
    radius = z_height  

    xy_offset = radius * math.cos(math.radians(angle))  
    z_offset = radius * math.sin(math.radians(angle))  

    cam1 = Vector((x + xy_offset, y, z_offset)) 
    cam2 = Vector((x - xy_offset, y, z_offset))  

    cam3 = Vector((x, y + xy_offset, z_offset))  
    cam4 = Vector((x, y - xy_offset, z_offset))  

    return [cam1, cam2, cam3, cam4]

def compute_lookat_rotation(cam_location, target_location, up_axis=Vector((0, 0, 1))):

    forward = (cam_location - Vector(target_location)).normalized()

    right = up_axis.cross(forward).normalized()

    up = forward.cross(right).normalized()

    rot_mat = Matrix((right, up, forward)).transposed()

    return rot_mat

    return rot_matrix

def create_persp_camera(cam_location, target_location, cam_name="PerspCam"):
    cam_data = bpy.data.cameras.new(name=cam_name)
    cam_data.type = 'PERSP'
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(120)

    cam_obj = bpy.data.objects.new(name=cam_name, object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = Vector(cam_location)

    rot_matrix = compute_lookat_rotation(cam_obj.location, Vector(target_location))
    cam_obj.rotation_euler = rot_matrix.to_euler()

    return cam_obj


def compute_dynamic_fov(cam_location, cam_target, bbox_corners, margin_ratio=1.05):

    cam_forward = (cam_target - cam_location).normalized()
    cam_right = cam_forward.cross(Vector((0,0,1))).normalized()
    cam_up = cam_right.cross(cam_forward).normalized()

    R = Matrix((cam_right, cam_up, cam_forward)).transposed()

    max_angle = 0

    for corner in bbox_corners:
        vec = corner - cam_location
        local_vec = R @ vec

        x = local_vec.x
        y = local_vec.y
        z = local_vec.z

        if z <= 0:
            continue

        horizontal_angle = math.atan(abs(x) / z)
        vertical_angle = math.atan(abs(y) / z)

        angle = max(horizontal_angle, vertical_angle)

        if angle > max_angle:
            max_angle = angle

    fov = 2 * max_angle * margin_ratio

    return fov


def get_obb_corners(obj):
    vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if len(vertices) == 0:
        return None
    points = np.array([[v.x, v.y, v.z] for v in vertices])
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    cov = np.cov(points_centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    obb_coords = points_centered @ eig_vecs
    min_obb = np.min(obb_coords, axis=0)
    max_obb = np.max(obb_coords, axis=0)
    obb_corners_local = np.array([
        [min_obb[0], min_obb[1], min_obb[2]],
        [min_obb[0], min_obb[1], max_obb[2]],
        [min_obb[0], max_obb[1], min_obb[2]],
        [min_obb[0], max_obb[1], max_obb[2]],
        [max_obb[0], min_obb[1], min_obb[2]],
        [max_obb[0], min_obb[1], max_obb[2]],
        [max_obb[0], max_obb[1], min_obb[2]],
        [max_obb[0], max_obb[1], max_obb[2]],
    ])
    obb_corners_world = obb_corners_local @ eig_vecs.T + centroid
    return obb_corners_world.tolist()

def get_world_aabb(obj):
    local_bbox_corners = [Vector(corner) for corner in obj.bound_box]
    world_bbox_corners = [obj.matrix_world @ corner for corner in local_bbox_corners]
    min_corner = Vector(map(min, zip(*world_bbox_corners)))
    max_corner = Vector(map(max, zip(*world_bbox_corners)))
    return [list(min_corner), list(max_corner)]

def compute_bbox(wall_objs, scale = 1.0):
    all_verts = []
    for obj in wall_objs:
        if obj.type == 'MESH':
            all_verts += [obj.matrix_world @ v.co for v in obj.data.vertices]

    if not all_verts:
        raise ValueError("No vertices found for room walls.")

    min_corner = Vector((min(v.x for v in all_verts), min(v.y for v in all_verts), min(v.z for v in all_verts)))
    max_corner = Vector((max(v.x for v in all_verts), max(v.y for v in all_verts), max(v.z for v in all_verts)))
    size = max_corner - min_corner
    center = (min_corner + max_corner) / 2
    if scale != 1.0:
        min_corner = center + (min_corner - center) * scale
        max_corner = center + (max_corner - center) * scale
        size = max_corner - min_corner
    return min_corner, max_corner, size, center

def compute_bbox_volume(min_corner, max_corner):
    length = max_corner.x - min_corner.x
    width = max_corner.y - min_corner.y
    height = max_corner.z - min_corner.z
    volume = length * width * height
    return volume
def compute_bbox_area(min_corner, max_corner):
    length = max_corner.x - min_corner.x
    width = max_corner.y - min_corner.y
    height = max_corner.z - min_corner.z
    area1 = length * width
    area2 = length * height
    area3 = width * height
    return area1, area2, area3

def top_ranked_objects_by_bbox_volume_area(objs, rank_num):
    heap_volume = []
    heap_area = []
    obj_info = {}  # record (volume, area)

    for obj in objs:
        name_low = obj.name.lower()
        if "door" in name_low or "window" in name_low:
            continue
        elif "table" not in name_low and "cabinet" not in name_low and "shelf" not in name_low and "bookcase" not in name_low:
            continue
        elif "trinkets" in name_low:
            continue
        try:
            min_corner, max_corner = compute_bbox([obj])[:2]
            volume = compute_bbox_volume(min_corner, max_corner)
            area = max(compute_bbox_area(min_corner, max_corner))
        except Exception as e:
            print(f"[Warning] Failed to compute bbox for {obj.name}: {e}")
            continue

        obj_info[obj] = (volume, area)

        if len(heap_volume) < rank_num:
            heapq.heappush(heap_volume, (volume, id(obj), obj))
        else:
            heapq.heappushpop(heap_volume, (volume, id(obj), obj))

        if len(heap_area) < rank_num:
            heapq.heappush(heap_area, (area, id(obj), obj))
        else:
            heapq.heappushpop(heap_area, (area, id(obj), obj))

    top_volume_objs = set(obj for _, __, obj in heap_volume)
    top_area_objs = set(obj for _, __, obj in heap_area)

    combined_objs = top_volume_objs.union(top_area_objs)

    result = []
    for obj in combined_objs:
        volume, area = obj_info[obj]
        result.append([obj, volume, area])

    result.sort(key=lambda x: (-x[1], -x[2]))
    return result

def add_captions_for_objects(objs_with_info, font_size=0.3, z_offset=10.0):
    """Add 3D captions to Blender scene for given objects"""
    created_captions = []

    for obj, volume, area in objs_with_info:
        min_corner, max_corner, _, center = compute_bbox([obj])

        object_height = max_corner.z - min_corner.z
        caption_z = max_corner.z + z_offset

        text_data = bpy.data.curves.new(name=f"Text_{obj.name}", type='FONT')
        text_data.body = obj.name.split("Factory")[0]
        text_data.size = font_size
        text_data.align_x = 'CENTER'
        text_data.align_y = 'CENTER'

        text_obj = bpy.data.objects.new(name=f"Caption_{obj.name}", object_data=text_data)
        text_obj.location = Vector((center.x, center.y, caption_z))
        text_obj.rotation_euler = (0, 0, 0) 
        text_obj.hide_render = False
        text_obj.hide_viewport = False

        bpy.context.collection.objects.link(text_obj)

        mat_name = "Caption_Red"
        if mat_name not in bpy.data.materials:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            for node in nodes:
                nodes.remove(node)
            
            emission = nodes.new(type='ShaderNodeEmission')
            emission.inputs[0].default_value = (1, 0, 0, 1) 
            emission.inputs[1].default_value = 5.0 
            
            output = nodes.new(type='ShaderNodeOutputMaterial')
            
            links.new(emission.outputs[0], output.inputs[0])
        else:
            mat = bpy.data.materials[mat_name]

        if text_obj.data.materials:
            text_obj.data.materials[0] = mat
        else:
            text_obj.data.materials.append(mat)

        created_captions.append(text_obj)

    return created_captions

def remove_all_captions():
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Caption_") and obj.type == 'FONT':
            bpy.data.objects.remove(obj, do_unlink=True)