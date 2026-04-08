import bpy
import os
from mathutils import Vector, geometry, Matrix
import bmesh
import math
import json
from PIL import Image
import cv2
from util import *

USE_SCENE_COPY = True
render_resolution = (512, 512)

def find_filled_floor_by_location():
    asset_loc = get_first_spawn_asset_location()
    floor_col = bpy.data.collections.get("unique_assets:room_floor")
    if not floor_col:
        raise ValueError("Collection 'unique_assets:room_floor' not found.")

    asset_point_2d = Vector((asset_loc.x, asset_loc.y))
    
    for floor_obj in floor_col.objects:
        if floor_obj.type != 'MESH':
            continue

        bm = bmesh.new()
        bm.from_mesh(floor_obj.data)
        bm.transform(floor_obj.matrix_world)
        
        up_vector = Vector((0, 0, 1))
        
        for face in bm.faces:
            if face.normal.dot(up_vector) > 0.9:
                verts_2d = [Vector((v.co.x, v.co.y)) for v in face.verts]

                if is_point_in_polygon(asset_point_2d, verts_2d):
                    print(f"Matched floor: {floor_obj.name}")
                    bm.free()
                    return floor_obj
        
        bm.free()
    
    closest_floor = None
    min_distance = float('inf')
    
    for floor_obj in floor_col.objects:
        if floor_obj.type != 'MESH':
            continue
        
        verts = [floor_obj.matrix_world @ v.co for v in floor_obj.data.vertices]
        center = sum(((v.x, v.y) for v in verts), ()) / (len(verts) * 2)
        center = Vector((center[0], center[1]))
        
        dist = (center - asset_point_2d).length
        
        if dist < min_distance:
            min_distance = dist
            closest_floor = floor_obj
    
    if closest_floor:
        print(f"No exact match found. Using closest floor: {closest_floor.name}")
        return closest_floor
        
    raise ValueError("No room floor contains the furniture location.")

def is_within_threshold(point, triangle_pts, threshold=1.0):
    pts = np.asarray(triangle_pts, dtype=float)  
    centroid = pts.mean(axis=0)                   

    pt = np.asarray(point, dtype=float)
    dist = np.linalg.norm(pt - centroid)
    return dist < threshold


def compute_floor_vertices(scene_path, output_dir):
    """Floor vert.co extract to json"""
    timestamp = scene_path.split("/")[-3]
    print(f"[Start] Reading {scene_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.wm.open_mainfile(filepath=scene_path)
    floor_obj = find_filled_floor_by_location()

        
    bm = bmesh.new()
    bm.from_mesh(floor_obj.data)
    bm.transform(floor_obj.matrix_world)

    vertex_data = []

    for idx, vert in enumerate(bm.verts, start=1):
        print(f"The {idx} vertice's coordinate is {vert.co}.")
        vertex_data.append({
            "idx": idx,
            "coordinate": [vert.co.x, vert.co.y, vert.co.z]
        })

    bm.free()

    os.makedirs(output_dir, exist_ok=True) 
    output_json_path = os.path.join(output_dir, f"{timestamp}.json")
    with open(output_json_path, 'w') as f:
        json.dump(vertex_data, f, indent=4)

    print(f"[Saved] Floor vertex data saved to {output_json_path}")


def hide_all_rooms_except_filled(filled_floor_obj):
    floor_name = filled_floor_obj.name
    room_prefix = floor_name.replace('.floor', '')
    print(f"[Hide] Room prefix extracted: {room_prefix}")
    
    hide_types = {"room_ceiling", "room_exterior", "room_wall"}

    for col in bpy.data.collections:
        # remove floating objects like ceiling light
        for obj in col.objects:
            obj_name = obj.name 
            if obj_name.startswith("CeilingLightFactory"):
                print(f"The hided object is {obj_name}")
                obj.hide_render = True
        col_name = col.name
        if col_name.lower() == "skirting":
            for obj in col.objects:
                obj.hide_render = True
            continue
        if not col_name.startswith("unique_assets:"):
            continue
        asset_type = col_name.split("unique_assets:")[-1]

        for obj in col.objects:
            if obj.name.startswith(room_prefix):
                obj.hide_render = asset_type in hide_types
            else:
                obj.hide_render = True

def get_objects_to_render_from_json(json_path, filled_floor_obj):
    names_to_render = set()

    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and "objs" in data:
        obj_list = data["objs"].values()
    elif isinstance(data, list):
        obj_list = data
    else:
        raise ValueError("Unsupported JSON structure.")

    for item in obj_list:
        obj_name = item.get("name")
        if obj_name:
            names_to_render.add(obj_name)

    names_to_render.add(filled_floor_obj.name)
    
    objects_to_render = []
    objects_room_to_render = []
    for name in names_to_render:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            # if "door" in name.lower() or "window" in name.lower():
            if "door" in name.lower():
                obj.hide_render = False
                obj.hide_viewport = False
            objects_room_to_render.append(obj)
        else:
            print(f"[Warning] Object {name} not found in scene!")
    for obj in objects_room_to_render:
        if not obj.name.endswith('.floor'):
            objects_to_render.append(obj)
    print(f"[Info] Found {len(objects_to_render)} objects to render.")
    print(f"[Info] Found {len(objects_room_to_render)} objects to render.")
    return objects_to_render, objects_room_to_render

def setup_topdown_camera(objects_to_render, filled_floor_obj, z_margin=20.0, safety_ratio=1.1):
    floor = filled_floor_obj.name
    floor_objs = [obj for obj in objects_to_render if obj.name.startswith(floor)]

    if not floor_objs:
        raise ValueError("No wall objects found among selected objects.")

    min_corner, max_corner, size, center = compute_bbox(floor_objs)

    move_objects_to_origin(objects_to_render, center)

    room_length = size.x
    room_width = size.y

    cam_data = bpy.data.cameras.new("TopDownCam")
    cam_data.type = 'ORTHO'
    ortho_scale = max(room_length, room_width) * safety_ratio
    cam_data.ortho_scale = ortho_scale

    cam_obj = bpy.data.objects.new("TopDownCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = (0, 0, z_margin)
    cam_obj.rotation_euler = (0, 0, 0)  

    bpy.context.scene.camera = cam_obj

    print(f"[Camera Setup] ortho_scale={ortho_scale:.2f}, height={cam_obj.location.z:.2f}")

    return cam_obj

def setup_oblique_cameras(objects_to_render, filled_floor_obj, distance_ratio=1.2, z_height=6.0, safety_ratio=1.1):
    floor_name = filled_floor_obj.name
    floor_objs = [o for o in objects_to_render if o.name.startswith(floor_name)]
    min_c, max_c, size, center = compute_bbox(floor_objs)

    move_objects_to_origin(objects_to_render, center)
    
    lx, ly = size.x, size.y
    diag = math.sqrt(lx**2 + ly**2) / 2 * distance_ratio
    offsets = [
        ( 0,  -diag/2, z_height),  
        # ( diag, -diag, z_height),  
        # (-diag, -diag, z_height),  
        # (-diag,  diag, z_height), 
    ]

    cams = []
    for i, (x, y, z) in enumerate(offsets):
        cam_obj = create_persp_camera((x,y,z), (0,0,0),cam_name=f"ObliqueCam_{i}")
        cams.append(cam_obj)
    return cams

def disable_all_lights():
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True

def setup_render_settings(output_path, resolution, samples, engine="CYCLES"):
    if engine == "CYCLES":
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences

        for ctype in ('CUDA', 'OPTIX', 'OPENCL'):
            try:
                cprefs.compute_device_type = ctype
                cprefs.get_devices()
            except Exception:
                continue
            else:
                break
        
        gpu_devices = [d for d in cprefs.devices if d.type != 'CPU']
        if gpu_devices:
            bpy.context.scene.cycles.device = 'GPU'
            for d in cprefs.devices:
                d.use = (d.type != 'CPU')
            print(f"[Info] Using GPU devices: {[d.name for d in gpu_devices]}")
        else:
            bpy.context.scene.cycles.device = 'CPU'
            for d in cprefs.devices:
                d.use = (d.type == 'CPU')
            print("[Info] No GPU found, using CPU only.")

    assert engine == "BLENDER_EEVEE_NEXT" or engine == "CYCLES"
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.view_settings.exposure = -2.0
    if engine == "CYCLES":
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_adaptive_sampling = True

    # bpy.context.scene.world.use_nodes = True
    # bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    # if bg_node:
    #     bg_node.inputs[0].default_value = (1, 1, 1, 1)
    #     bg_node.inputs[1].default_value = 0 

def filter_and_save_scene_by_room_prefix(filled_floor_obj):
    floor_name = filled_floor_obj.name
    room_prefix = floor_name.replace('.floor', '')
    print(f"[Filter] Extracted room prefix: {room_prefix}")

    delete_types = {"room_ceiling", "room_exterior", "doors", "windows"}
    for col in bpy.data.collections:
        for obj in col.objects:
            obj_name = obj.name 
            if obj_name.startswith("CeilingLightFactory"):
                print(f"The removed object is {obj_name}")
                bpy.data.objects.remove(obj, do_unlink=True)
        col_name = col.name
        if col_name.lower() == "skirting":
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            continue

        if not col_name.startswith("unique_assets:"):
            continue

        asset_type = col_name.split("unique_assets:")[-1]

        for obj in list(col.objects):
            if obj.name.startswith(room_prefix):
                if asset_type in delete_types:
                    bpy.data.objects.remove(obj, do_unlink=True)
            else:
                bpy.data.objects.remove(obj, do_unlink=True)

    input_folder = os.path.dirname(scene_path)
    output_blend = os.path.join(input_folder, "filtered.scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    print(f"Saved filtered scene to: {output_blend}")

def setup_single_object_camera(angle_deg=0):
    cam_data = bpy.data.cameras.new(name="ObjectCam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = 2.0

    cam_obj = bpy.data.objects.new(name="ObjectCam", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)

    radius = 2.0
    angle_rad = math.radians(angle_deg)

    cam_obj.location = (radius * math.sin(angle_rad), -radius * math.cos(angle_rad), 0)
    cam_obj.rotation_euler = (math.radians(90), 0, angle_rad)

    bpy.context.scene.camera = cam_obj
    return cam_obj

def render_single_object(obj, output_dir, resolution=(512, 512), samples=64):
    obj_copy = obj.copy()
    obj_copy.data = obj.data.copy()
    bpy.context.collection.objects.link(obj_copy)

    for o in bpy.data.objects:
        o.hide_render = True
    obj_copy.hide_render = False

    normalize_object(obj_copy)

    angles = [0, 45, -45]

    for angle in angles:
        cam_obj = setup_single_object_camera(angle)

        bpy.context.scene.render.engine = 'CYCLES'
        output_file = os.path.join(output_dir, f"{obj.name}_view{angle:+d}.png")
        bpy.context.scene.render.filepath = output_file
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_adaptive_sampling = True

        bpy.context.scene.world.use_nodes = True
        bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs[0].default_value = (1, 1, 1, 1)
            bg_node.inputs[1].default_value = 5.0 

        bpy.ops.render.render(write_still=True)
        print(f"Rendered: {obj.name} at {angle:+d}°")

        bpy.data.objects.remove(cam_obj, do_unlink=True)

    bpy.data.objects.remove(obj_copy, do_unlink=True)

def render_main_object_views(obj, oid, objects_filtered, output_dir, margin=1.05, samples=512, check_existence=True):
    objects_to_cleanup = []
    try:
        obj_copy = obj.copy()
        if obj.data:
            obj_copy.data = obj.data.copy()
        bpy.context.collection.objects.link(obj_copy)
        objects_to_move = [obj_copy]
        objects_to_cleanup.append(obj_copy)
    
        min_corner, max_corner, _, center = compute_bbox([obj_copy])
        move_vector = center - Vector((0, 0, 0))
        
        bm = bmesh.new()
        bm.from_mesh(obj_copy.data)
        bm.transform(obj_copy.matrix_world)
        
        related_objects = []
        for o in objects_filtered:                
            try:
                o_copy = o.copy()
                if o.data:
                    o_copy.data = o.data.copy()
                bpy.context.collection.objects.link(o_copy)
                
                for col in o_copy.users_collection:
                    if o_copy.name!=obj.name and col.name!="unique_assets:doors" and col.name!="unique_assets:windows":
                        _, _, _, center = compute_bbox([o_copy])
                        point = (center.x, center.y)
                        for face in bm.faces:
                            verts_2d = [Vector((v.co.x, v.co.y)) for v in face.verts]
                            if is_point_in_polygon(point, verts_2d):
                                related_objects.append(o_copy)
                                break
            except Exception as e:
                print(f"[Warning] Failed to process object {o.name}: {e}")
        
        bm.free()

        objects_to_move.extend(related_objects)
        objects_to_cleanup.extend(related_objects)
        
        for o in objects_to_move:
            o.hide_render = False 
            
        move_objects_to_origin(objects_to_move, move_vector)
        
        min_corner, max_corner, size, center = compute_bbox(objects_to_move)

        views = {
            'X': Vector((1, 0, 0)),
            'Y': Vector((0, 1, 0)),
            'IX': Vector((-1, 0, 0)),
            'IY': Vector((0, -1, 0)),
            'Z': Vector((0, 0, 1)),
        }

        setup_render_settings("", render_resolution, samples)
        
        # Render from each view
        for axis, dir_vec in views.items():
            try:
                # Calculate plane size based on axis
                if axis == 'X' or axis == 'IX':
                    plane_size = max(size.y, size.z)
                elif axis == 'Y' or axis == 'IY':
                    plane_size = max(size.x, size.z)
                else:
                    plane_size = max(size.x, size.y)
                    
                ortho_scale = plane_size * margin

                # Create and setup camera
                cam_data = bpy.data.cameras.new(f"OrthoCam_{axis}_{obj_copy.name}")
                cam_data.type = 'ORTHO'
                cam_data.ortho_scale = ortho_scale
                cam_obj = bpy.data.objects.new(cam_data.name, cam_data)
                objects_to_cleanup.append(cam_obj)

                # Position camera
                cam_obj.location = center + dir_vec * (plane_size * 2)
                direction = center - cam_obj.location
                rot_quat = direction.to_track_quat('-Z', 'Y')
                cam_obj.rotation_euler = rot_quat.to_euler()

                bpy.context.collection.objects.link(cam_obj)
                bpy.context.scene.camera = cam_obj

                # Setup output path
                os.makedirs(output_dir, exist_ok=True)
                name = obj_copy.name.split("Factory")[0]
                
                base_path = os.path.join(output_dir, f"{axis}_{name}.png")
                
                # Check if file exists and handle accordingly
                if check_existence and os.path.exists(base_path):
                    idx = 1
                    while os.path.exists(os.path.join(output_dir, f"{axis}_{name}_{idx}.png")):
                        idx += 1
                    path = os.path.join(output_dir, f"{axis}_{name}_{idx}.png")
                else:
                    path = base_path
                    
                bpy.context.scene.render.filepath = path
                print(f"[Render] {path}")
                render_scene()
                rendered_img = cv2.imread(path)
                cv2.putText(rendered_img, f"{oid+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(path.replace(".png", "_annotated.png"), rendered_img)
                
            except Exception as e:
                print(f"[Warning] Failed to render {axis} view: {e}")
                
    except Exception as e:
        print(f"[Warning] Failed to process main object {obj.name}: {e}")
    finally:
        # Cleanup all created objects
        for obj_to_clean in objects_to_cleanup:
            try:
                if obj_to_clean and obj_to_clean.name in bpy.data.objects:
                    bpy.data.objects.remove(obj_to_clean, do_unlink=True)
            except Exception as e:
                print(f"[Warning] Failed to cleanup object {obj_to_clean.name if obj_to_clean else 'unknown'}: {e}")

def concat_images(paths, output_path):
    """Concatenate images n*5 grid."""
    images = [cv2.imread(path) for path in paths]
    images_columns = []
    for idx in range(0, len(images), 5):
        images_row = images[idx:idx+5]
        image_row = np.concatenate(images_row, axis=1)
        images_columns.append(image_row)
    image_concat = np.concatenate(images_columns, axis=0)
    cv2.imwrite(output_path, image_concat)
    
def render_topdown(scene_path, output_path, json_path, render_main_objects=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    try:
        bpy.ops.wm.open_mainfile(filepath=scene_path)

        filled_floor = find_filled_floor_by_location()
        hide_all_rooms_except_filled(filled_floor)

        objects_filtered, objects_room_filtered = get_objects_to_render_from_json(json_path, filled_floor)
        main_objects = top_ranked_objects_by_bbox_volume_area(objects_filtered, rank_num=5)
        floor_objs = [filled_floor]
        if not floor_objs:
            raise ValueError("No wall objects found among selected objects.")
        min_corner, max_corner, size, center = compute_bbox(floor_objs)
        room_length = size.x
        room_width = size.y
        room_scale = max(room_length, room_width)

        bbox_2d_main_objs = []
        for obj in main_objects:
            obj = obj[0]
            min_corner, max_corner = compute_bbox([obj])[:2]
            min_corner = min_corner-center
            max_corner = max_corner-center
            inv_safety_ratio = 0.90
            bbox = [((min_corner[0]/room_scale*inv_safety_ratio)+0.5)*render_resolution[0], 
                    ((-min_corner[1]/room_scale*inv_safety_ratio)+0.5)*render_resolution[1],
                    ((max_corner[0]/room_scale*inv_safety_ratio)+0.5)*render_resolution[0], 
                    ((-max_corner[1]/room_scale*inv_safety_ratio)+0.5)*render_resolution[1]]
            # scale the bbox to 1.2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            bbox = [cx - w * 0.6, cy - h * 0.6, cx + w * 0.6, cy + h * 0.6]
            bbox_2d_main_objs.append([obj.name, bbox])
        
        # import pdb; pdb.set_trace()
        captions = add_captions_for_objects(main_objects, font_size=0.3)
        if captions is None:
            print("ERRRRRRRRROR!!!!!")
            return
        else:
            idx=1
            for caption in captions:
                print(f"The {idx} caption is {caption}")
        print(f"The top {len(main_objects)} main objects are {main_objects}")
        
        objects_to_cleanup = []
        
        # Check if main_objects directory exists
        main_objects_dir = os.path.join(os.path.dirname(output_path), "main_objects")
        should_render_main_objects = render_main_objects and not os.path.exists(main_objects_dir)
        # should_render_main_objects = True
        
        if USE_SCENE_COPY:
            print("[Info] Using copied objects for rendering.")
            copied_objects = []
            for obj in objects_room_filtered:
                try:
                    obj_copy = obj.copy()
                    if obj.data:
                        obj_copy.data = obj.data.copy()
                    bpy.context.collection.objects.link(obj_copy)
                    copied_objects.append(obj_copy)
                    objects_to_cleanup.append(obj_copy)
                except Exception as e:
                    print(f"[Warning] Failed to copy object {obj.name}: {e}")
            objects_to_render = copied_objects + captions
        else:
            print("[Info] Using original objects for rendering.")
            objects_to_render = objects_room_filtered + captions

        # Check if both topdown and annotated files exist
        annotated_path = output_path.replace("topdown", "topdown_annotated")
        need_render = not os.path.exists(output_path)
        need_annotated = not os.path.exists(annotated_path)
        
        if need_render or need_annotated:
            # Show all objects and captions for topdown view
            for obj in objects_to_render:
                obj.hide_render = False
            
            for obj in objects_room_filtered:
                obj.hide_render = True
            for caption in captions:
                caption.hide_render = True
                caption.hide_viewport = False

            topdown_camera = setup_topdown_camera(objects_to_render, filled_floor)
            objects_to_cleanup.append(topdown_camera)
            
            setup_render_settings(output_path, render_resolution, 256)

            bpy.context.scene.render.filepath = output_path
            bpy.context.scene.camera = topdown_camera
            
            if need_render:
                render_scene()
                print(f"[Done] Saved render to {output_path}")
            else:
                print(f"[Skip] Topdown render already exists: {output_path}")
            
            # Generate annotated version if needed
            if need_annotated:
                print(f"[Generating] Creating annotated version: {annotated_path}")
                rendered_img = cv2.imread(output_path)
                for bbox_id, bbox in enumerate(bbox_2d_main_objs):
                    cv2.rectangle(rendered_img, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[1][2]), int(bbox[1][3])), (0, 0, 255), 2)
                    cv2.putText(rendered_img, f"{bbox_id+1}", (int(bbox[1][0])+5, int(bbox[1][1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(annotated_path, rendered_img)
                print(f"[Done] Saved annotated render to {annotated_path}")
            else:
                print(f"[Skip] Annotated render already exists: {annotated_path}")
        else:
            print(f"[Skip] Both topdown and annotated renders already exist")
        
        remove_all_captions()

        # Render main objects if needed
        if should_render_main_objects:
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    obj.hide_render = True

            os.makedirs(main_objects_dir, exist_ok=True)
            
            for oid, (obj, _, _) in enumerate(main_objects):
                render_main_object_views(
                    obj=obj,
                    oid=oid,
                    objects_filtered=objects_filtered,
                    output_dir=main_objects_dir,
                    margin=1.05,
                    samples=256,
                    check_existence=True
                )
        else:
            print(f"[Skip] Main objects renders already exist in: {main_objects_dir}")
                    
        # cancat all images
        obj_images_path = [os.path.join(main_objects_dir, obj_img_name) for obj_img_name in os.listdir(main_objects_dir) if obj_img_name.endswith("_annotated.png")]
        concat_images(obj_images_path, os.path.join(main_objects_dir, f"render_main_objects_{scene_path.split('/')[-3]}.png"))
    except Exception as e:
        print(f"[Error] Failed to render scene: {e}")
        raise
    finally:
        for obj in objects_to_cleanup:
            try:
                if obj and obj.name in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            except Exception as e:
                print(f"[Warning] Failed to cleanup object {obj.name if obj else 'unknown'}: {e}")

def render_oblique(scene_path, output_dir, json_path, use_scene_copy=False):
    all_exist = all(os.path.exists(f"{output_dir}_oblique_{i}.png") for i in range(4))
    if all_exist:
        print(f"[Skip] Oblique renders already exist: {output_dir}")
        return

    print(f"[Start] Rendering oblique views for {scene_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=scene_path)

    filled_floor = find_filled_floor_by_location()
    hide_all_rooms_except_filled(filled_floor)
    objs_all, objs_room = get_objects_to_render_from_json(json_path, filled_floor)

    if use_scene_copy:
        print("[Info] Using copied objects for rendering.")
        objs_to_render = []
        for obj in objs_room:
            c = obj.copy()
            if obj.data:
                c.data = obj.data.copy()
            bpy.context.collection.objects.link(c)
            objs_to_render.append(c)
            obj.hide_render = True
    else:
        print("[Info] Using original objects for rendering.")
        objs_to_render = objs_room

    cams = setup_oblique_cameras(objs_to_render, filled_floor, distance_ratio=1.2, z_height=6.0)

    setup_render_settings(output_dir, render_resolution, 512)
    # setup_render_settings(output_dir, render_resolution, 1024, "BLENDER_EEVEE_NEXT")

    for idx, cam in enumerate(cams):
        bpy.context.scene.camera = cam
        out_path = f"{output_dir}_oblique_{idx}.png"
        bpy.context.scene.render.filepath = out_path
        print(f"[Render] {out_path}")
        render_scene()
        bpy.data.objects.remove(cam, do_unlink=True)

    if use_scene_copy:
        cleanup_copied_objects(objs_to_render)

    print(f"[Done] Saved 4 oblique renders to {output_dir}")

def render_floormask(scene_path, output_dir):
    print(f"[Start] Floor mask rendering from {scene_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=scene_path)

    timestamp = scene_path.split("/")[-3]
    output_path = os.path.join(output_dir, timestamp, f"floormask_{timestamp}.png")
    
    if os.path.exists(output_path):
        print(f"[Skip] Floor mask already exists: {output_path}")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    floor_obj = find_filled_floor_by_location()
    
    mat = bpy.data.materials.new(name="WhiteMaskMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[0].default_value = (1, 1, 1, 1) 
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs[0], output.inputs[0])
    floor_obj.active_material = mat

    wall_and_floor = [floor_obj]
    room_prefix = floor_obj.name.replace('.floor', '')
    for obj in bpy.data.objects:
        if obj.name!=floor_obj.name:
            obj.hide_render = True
        if obj.name == f"{room_prefix}.wall":
            wall_and_floor.append(obj)
            
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'BW' 
    bpy.context.scene.render.film_transparent = False  

    scene = bpy.context.scene
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0, 0, 0, 1)  

    topdown_camera = setup_topdown_camera(wall_and_floor, floor_obj)
    scene.camera = topdown_camera
    setup_render_settings(output_path, (1024,1024), 1)

    render_scene()
    bpy.data.objects.remove(topdown_camera, do_unlink=True)

def render_individual_objects_from_json(json_path, output_dir, resolution=(512, 512), samples=64):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and "objs" in data:
        obj_list = data["objs"].values()
    elif isinstance(data, list):
        obj_list = data
    else:
        raise ValueError("Unsupported JSON structure.")

    render_count = 0
    rank_num = 5
    ranked_objs = []
    for item in obj_list:
        obj_name = item.get("name")
        if not obj_name:
            continue

        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            print(f"[Warning] Object {obj_name} not found in scene.")
            continue
        
        render_single_object(obj, output_dir, resolution, samples)
        render_count += 1

        ranked_objs.append(obj)
        main_objects = top_ranked_objects_by_bbox_volume_area(ranked_objs, rank_num)
        add_captions_for_objects(main_objects, font_size=0.3)
       
    print(f"The top {len(main_objects)} main objects are {main_objects}")
    print(f"Total rendered objects: {render_count}")

def set_camera_pose_blender(cam_obj, radius, elevation, azimuth, center):
    x = center[0] - radius * np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(azimuth))
    y = center[1] - radius * np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth))
    z = center[2] - radius * np.sin(np.deg2rad(elevation))
    cam_obj.location = (x, y, z)
    direction = Vector(center) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

def setup_gif_camera(img_width = 1100, img_height = 700, focal_px = 1300):
    cx = img_width / 2
    cy = img_height / 2

    sensor_width_mm = 36.0

    cam_data = bpy.data.cameras.new(name="GifCam")
    cam_data.type = 'PERSP'
    cam_data.lens_unit = 'MILLIMETERS'
    cam_data.sensor_fit = 'HORIZONTAL'

    cam_obj = bpy.data.objects.new(name="GifCam", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)

    scene = bpy.context.scene
    scene.render.resolution_x = img_width
    scene.render.resolution_y = img_height
    scene.render.resolution_percentage = 100
    
    cam_data.lens = focal_px * sensor_width_mm / img_width

    cam_data.shift_x = (cx - img_width * 0.5) / img_width
    cam_data.shift_y = -(cy - img_height * 0.5) / img_height

    return cam_obj


def render_gif(scene_path, output_dir, json_path, use_scene_copy=False, radius=10, elevation=-10, start_azimuth=-40, end_azimuth=40, step=4, duration=100):
    print(f"[Start] Rendering frames for {scene_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=scene_path)

    filled_floor = find_filled_floor_by_location()
    hide_all_rooms_except_filled(filled_floor)
    objs_all, objs_room = get_objects_to_render_from_json(json_path, filled_floor)

    # For bedroom's objects on bed, specially put under Collection directly, as a result with a different filtering method
    extra_kw = (
        "blanketfactory",
        "comforterfactory",
        "mattressfactory",
        "pillowfactory",
        "towelfactory",
    )

    top_col = bpy.data.collections.get("Collection")
    if top_col is None:
        raise RuntimeError('No top "Collection", check the level!')

    def all_objects_recursive(col):
        objs = set(col.objects)
        for child in col.children:
            objs |= all_objects_recursive(child)
        return objs

    top_objs = all_objects_recursive(top_col)

    added = 0
    for obj in top_objs:
        if any(kw in obj.name.lower() for kw in extra_kw):
            if obj not in objs_room:
                objs_room.append(obj)
                added += 1

    print(f"[Info] Supplement from the top Collection, Add {added}'s mattress, pillow and other objects")
    
    if use_scene_copy:
        print("[Info] Using copied objects for rendering.")
        objs_to_render = []
        for obj in objs_room:
            c = obj.copy()
            if obj.data:
                c.data = obj.data.copy()
            bpy.context.collection.objects.link(c)
            objs_to_render.append(c)
            obj.hide_render = True
    else:
        print("[Info] Using original objects for rendering.")
        objs_to_render = objs_room
    
    
    floor = filled_floor.name
    floor_objs = [obj for obj in objs_to_render if obj.name.startswith(floor)]
    min_c, max_c, size, center = compute_bbox(floor_objs)

    move_objects_to_origin(objs_to_render, center)
    normalize_for_gif(objs_room, target_size=5.0)

    _, _, scene_size, scene_center = compute_bbox(objs_to_render)
    radius = max(scene_size) * 2.0

    cam = setup_gif_camera(img_width = 1100, img_height = 700, focal_px = 1300)
    bpy.context.scene.camera = cam
    
    bpy.context.scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences

    for ctype in ('CUDA', 'OPTIX', 'OPENCL'):
        try:
            cprefs.compute_device_type = ctype
            cprefs.get_devices()
        except Exception:
            continue
        else:
            break
    
    gpu_devices = [d for d in cprefs.devices if d.type != 'CPU']
    if gpu_devices:
        bpy.context.scene.cycles.device = 'GPU'
        for d in cprefs.devices:
            d.use = (d.type != 'CPU')
        print(f"[Info] Using GPU devices: {[d.name for d in gpu_devices]}")
    else:
        bpy.context.scene.cycles.device = 'CPU'
        for d in cprefs.devices:
            d.use = (d.type == 'CPU')
        print("[Info] No GPU found, using CPU only.")
    
    bpy.context.scene.cycles.samples = 2048
    bpy.context.scene.cycles.use_adaptive_sampling = True
    
    frame_paths = []

    for azimuth in np.arange(start_azimuth, end_azimuth, step):
        set_camera_pose_blender(cam, radius, elevation, azimuth, scene_center)
        frame_path = os.path.join(output_dir, f"frame_{azimuth:.1f}.png")
        bpy.context.scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)
        frame_paths.append(frame_path)

    for azimuth in np.arange(end_azimuth, start_azimuth, -step):
        set_camera_pose_blender(cam, radius, elevation, azimuth, scene_center)
        frame_path = os.path.join(output_dir, f"frame_{azimuth:.1f}_reverse.png")
        bpy.context.scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)
        frame_paths.append(frame_path)

    frames = [Image.open(p) for p in frame_paths]
    gif_path = os.path.join(output_dir, "scene.gif")
    frames[0].save(gif_path, format="GIF", append_images=frames[1:], save_all=True, duration=duration, loop=0)
    print(f"[Done] Saved GIF to {gif_path}")

    # 关闭所有帧图片并删除中间文件
    for frame in frames:
        frame.close()
    for frame_path in frame_paths:
        if os.path.exists(frame_path):
            os.remove(frame_path)
    print(f"[Cleanup] Removed {len(frame_paths)} intermediate frame files")

    if use_scene_copy:
        cleanup_copied_objects(objs_to_render)



