import open3d as o3d
import trimesh
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uuid
from collections import defaultdict
import glob
import argparse
import shutil
import ast
# from make_mask import screenshot, make_mask
import cv2

# region_names = {'a': 'bathroom (should have a toilet and a sink)',
# 'b' : 'bedroom',
# 'c' : 'closet',
# 'd' : 'dining room (includes “breakfast rooms” other rooms people mainly eat in)',
# 'e' : 'entryway/foyer/lobby (should be the front door, not any door)',
# 'f' : 'familyroom (should be a room that a family hangs out in, not any area with couches)',
# 'g' : 'garage',
# 'h' : 'hallway',
# 'i' : 'library (should be room like a library at a university, not an individual study)',
# 'j' : 'laundryroom/mudroom (place where people do laundry, etc.)',
# 'k' : 'kitchen',
# 'l' : 'living room (should be the main “showcase” living room in a house, not any area with couches)',
# 'm' : 'meetingroom/conferenceroom',
# 'n' : 'lounge (any area where people relax in comfy chairs/couches that is not the family room or living room',
# 'o' : 'office (usually for an individual, or a small set of people)',
# 'p' : 'porch/terrace/deck/driveway (must be outdoors on ground level)',
# 'r' : 'rec/game (should have recreational objects, like pool table, etc.)',
# 's' : 'stairs',
# 't' : 'toilet (should be a small room with ONLY a toilet)',
# 'u' : 'utilityroom/toolroom',
# 'v' : 'tv (must have theater-style seating)',
# 'w' : 'workout/gym/exercise',
# 'x' : 'outdoor areas containing grass, plants, bushes, trees, etc.',
# 'y' : 'balcony (must be outside and must not be on ground floor)',
# 'z' : 'other room (it is clearly a room, but the function is not clear)',
# 'B' : 'bar',
# 'C' : 'classroom',
# 'D' : 'dining booth',
# 'S' : 'spa/sauna',
# 'Z' : 'junk (reflections of mirrors, random points floating in space, etc.)',
# '-' : 'no label'}

region_names = {'a': 'bathroom',
'b' : 'bedroom',
'c' : 'closet',
'd' : 'dining room',
'e' : 'entryway/foyer/lobby',
'f' : 'familyroom',
'g' : 'garage',
'h' : 'hallway',
'i' : 'library',
'j' : 'laundryroom/mudroom',
'k' : 'kitchen',
'l' : 'living room',
'm' : 'meetingroom/conferenceroom',
'n' : 'lounge',
'o' : 'office',
'p' : 'porch/terrace/deck/driveway',
'r' : 'rec/game',
's' : 'stairs',
't' : 'toilet',
'u' : 'utilityroom/toolroom',
'v' : 'tv (must have theater style seating)',
'w' : 'workout/gym/exercise',
'x' : 'outdoor areas containing grass, plants, bushes, trees, etc',
'y' : 'balcony',
'z' : 'other room (it is clearly a room, but the function is not clear)',
'B' : 'bar',
'C' : 'classroom',
'D' : 'dining booth',
'S' : 'spa/sauna',
'Z' : 'junk',
'-' : 'no label'}


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


def screenshot(mesh):
    
    width, height = 256, 256  # 1024, 1024   # 3840, 2160  #1920, 1080
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])  # White background

    # scene.scene.set_render_option("msaa", 4)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    material.point_size = 3.0
    scene.add_geometry("mesh", mesh, material)

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    cam_pos = center + np.array([0, 0, extent[2]*2])  
    up = [0, 1, 0]  # up Y
    scene.camera.look_at(center, cam_pos, up)

    image = renderer.render_to_image()

    return image
    
def make_mask(last_save_path, threshold=230):
    # # Convert top-view image to mask
    image = cv2.imread(last_save_path) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    for i in range(h):
        for j in range(w):
            if gray[i][j] <= threshold:
                gray[i][j] = 0
            else:
                gray[i][j] = 1

    mask = gray
    mask_img = (mask * 255).astype(np.uint8)

    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(last_save_path, ~closed)

def obb_to_rotated_bbox(centroid, axes_lengths, normalized_axes):
    # # Convert OBB to (center, size, yaw) representation
    centroid = np.asarray(centroid)
    lengths = np.asarray(axes_lengths)
    axes = np.asarray(normalized_axes)

    # # Find the principal axis most aligned with the Z axis
    z_axis = np.array([0, 0, 1])
    projections = np.abs(axes.T @ z_axis)
    vert_idx = np.argmax(projections)

    dz = lengths[vert_idx] 

    # Find the two horizontal principal axes
    horz_idxs = [i for i in range(3) if i != vert_idx]
    ax1, ax2 = axes[:, horz_idxs[0]], axes[:, horz_idxs[1]]
    len1, len2 = lengths[horz_idxs[0]], lengths[horz_idxs[1]]

    def xy_proj(vec):
        return np.linalg.norm([vec[0], vec[1]])

    if xy_proj(ax1) > xy_proj(ax2):
        forward = ax1     
        dx, dy = len1, len2
    else:
        forward = ax2
        dx, dy = len2, len1

    yaw = np.arctan2(forward[1], forward[0]) 

    size = [dx, dy, dz]
    return centroid.tolist(), size, yaw

def draw_bounding_box(centroid, axesLengths, normalizedAxes_XY, color=[1, 0, 0]):
    x = np.array(normalizedAxes_XY[:3])
    y = np.array(normalizedAxes_XY[3:])
    z = np.cross(x, y)
    axes = np.vstack([x, y, z]).T

    
    unit_box = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ])
 
    centroid = np.array(centroid)   # bbx中心点
    axesLengths = np.array(axesLengths) #bbx长宽高
    #axes = np.array(normalizedAxes).reshape(3, 3).T   #旋转矩阵，bbx朝向


    box_scaled = unit_box * axesLengths
    box_rotated = box_scaled @ axes.T
    box_vertices = box_rotated + centroid

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(box_vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    #return line_set
    return box_vertices


def compute_bbx(centroid, axesLengths, axes):
    # Compute the 8 corners of the bounding box
    
    unit_box = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ])
 
    centroid = np.array(centroid)   # centering
    axesLengths = np.array(axesLengths) # size
   

    box_scaled = unit_box * axesLengths
    box_rotated = box_scaled @ axes
    box_vertices = box_rotated + centroid

    return box_vertices

def test_error(args, info):
    regions_ply = glob.glob(os.path.join(args.path_region_ply, '*.ply')) 
    for region_path in regions_ply:
        mesh = o3d.io.read_triangle_mesh(region_path)
        region_id = int(region_path.split('/')[-1].replace('region', '').replace('.ply', ''))  # 提取region id， 0
        
        for obj in info[region_id]:
            print('区域，物体：', (obj['region_name_id'], obj['region_name'],obj['obj_cat_id'], obj['obj_name']))
            centroid = obj['centroid']
            axes = obj['axes']
            axesLengths = obj['axesLengths']
            eightPoints = obj['eightPoints'] 

            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]


            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(eightPoints)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
            
            if region_id == 0:
                print("centroid:", centroid)
                o3d.visualization.draw_geometries([mesh, line_set])

            

def read_from_house(args, lines, cat_name_list):
    region_cats = {}
    for line in lines:
        line = line.strip().split()
        obj = line[0]
        if obj == 'R': # "R" indicates a region/room
            region_id = line[1]
            region_clas = line[5]
            region_cats.update({region_id:region_clas})

    # matterport 3D provide two category mapping:(1) category_mapping.tsv -- https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv ;
    # (2)mpcat40.tsv -- https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
    # reference: https://github.com/niessner/Matterport/blob/master/data_organization.md
    region_mpcat40 = {}
    for line in lines:
        line = line.strip().split()
        obj = line[0]
        if obj == 'C': # "C" indicates category mapping for objects
            region_id = line[2]  
            mpcat40_region_id = line[4]  
            mpcat40_region_clas = line[5]  # mpcat40 mapping
            if '_' in mpcat40_region_clas:
               mpcat40_region_clas = mpcat40_region_clas.replace('_', ' ')
            region_mpcat40.update({region_id:mpcat40_region_clas})


    obj_info = {}
    for line in lines:
        line = line.strip().split()
        obj = line[0]
        if obj == 'O': # "O" indicates objects
            region_cat = int(line[2])  # Current region/room category

            # if region_cat not in [8]:  #选择哪个区域,debug 
            #     continue

            if region_cat == -1:  # Remove objects with region/room ID = -1 (ambiguous assignment
                continue

            if int(line[3]) != -1:
                obj_cat_id = int(line[3]) + 1  # object category
            else:
                obj_cat_id = int(line[3])

           
            if obj_cat_id in [-1]:  # remove "unknown"
                continue

            centroid = np.array([float(x) for x in line[4:7]])     
            normalizedAxes_XY = [float(x) for x in line[7:13]]     
            axesLengths = np.array([float(x) * 2  for x in line[13:16]])   

            region_key = region_cats[line[2]]
            region_name = region_names[region_key]  
            # print("region_name:", region_name)

            x = np.array(normalizedAxes_XY[:3])
            y = np.array(normalizedAxes_XY[3:])
            z = np.cross(x, y)
            axes = np.vstack([x, y, z])

            eightPoints = compute_bbx(centroid, axesLengths, axes)  

            _, size, yaw = obb_to_rotated_bbox(centroid, axesLengths, axes) 



            which_cat_name = region_mpcat40[str(obj_cat_id)]
            
            # Remove redundant categories
            if which_cat_name == 'door' and cat_name_list[obj_cat_id+1] in ['door handle', 'garage door opener', 
                                                                            'door knob', 'doorknob', 
                                                                          'door hinge', 'doorpost', 'door locker handle', 
                                                                          'ceiling door', 'closet door knob',
                                                                          'door outside', 'door']:
                continue
            
            # Remove redundant categories
            if which_cat_name == 'window' and cat_name_list[obj_cat_id+1] in ['window', 'window outside']:
                continue
            # if which_cat_name == 'door' and cat_name_list[obj_cat_id+1] not in  ['door frame', 'doorframe', 'frame for door',
            #                                                               'window or door']:
            #     continue
            
            # Remove redundant categories in "mpcat40"
            if which_cat_name == 'void' or which_cat_name == 'unlabeled':  
                continue
            # Remove redundant categories
            if which_cat_name == 'objects' or which_cat_name == 'misc':
                which_cat_name = cat_name_list[obj_cat_id+1]
                if which_cat_name in ['unknown', 'door stopper', 'window reflection', 'window frame reflection', 'cabinet door', 'unknown-hot tub',
                                     'projector opening', 'cleaning clutter', 'unknown remove', 'shelf side', 'paper storage', 'edge', 'unknown remove',
                                     'three', 'window closet door', 'archway corner', 'box opening', 'planter curb', 'unknown other room', 
                                     'hoses for chemical tank', 'closet mirror wall', 'recessed wall', 'garage door opener motor', 
                                     'garage door opener bar', 'garage door opener', 'ceiling wall', 'wall detail', 'wall pack', 'garage door motor',
                                     'door window', 'unknown other room', 'toilet seat liner dispenser', 'risers for theater seating', 'chair rail',
                                     'chair part', 'bath wall', 'cabinet door', 'falling light', 'ceiling light fixture connection', 'counter door',
                                     'kitchen counter support', 'stack', 'sitting area', 'overhang', 'dirt ground', 'top', 'hose outlet', 'closest area',
                                     'show space', 'access area', 'under stair', 'channel', 'cubicle', 'stone circle', 'yard', 'stall', 'unlabeled' ]:
                    continue
    

            infos = {'region_name_id': region_cat, 'region_name': region_name, 'obj_cat_id': None, 
                     'obj_name': which_cat_name, 'centroid': centroid, 'axes': axes, 'axesLengths': axesLengths, 'eightPoints': eightPoints,
                     'axisSize':size, 'rotation':yaw}

            if region_cat not in obj_info:
                obj_info[region_cat] = []
            
            obj_info[region_cat].append(infos)
    
    # test_error(args, obj_info)

    # Compute the number of regions and objects
    out_obj_info = {}
    region_i = {}  
    for region_idnum in obj_info:
        obj_j = {}  
        tmp = []

        cur_region_name = obj_info[region_idnum][0]['region_name']
        if cur_region_name not in region_i:
            region_i[cur_region_name] = 0
        region_i[cur_region_name] = region_i[cur_region_name] + 1
            
        for obj_jj in obj_info[region_idnum]:
            cur_obj_name = obj_jj['obj_name']
            if cur_obj_name not in obj_j:
                obj_j[cur_obj_name] = 0
            obj_j[cur_obj_name] = obj_j[cur_obj_name] + 1

            tmp.append({
                'region_name_id': obj_jj['region_name_id'], 'region_name': f"{obj_jj['region_name']}-{region_i[cur_region_name]}", 
                'obj_cat_id': obj_jj['obj_cat_id'], 'obj_name': f"{obj_jj['obj_name']}-{obj_j[cur_obj_name]}", 'centroid': obj_jj['centroid'], 
                'axes': obj_jj['axes'], 'axesLengths': obj_jj['axesLengths'], 'eightPoints': obj_jj['eightPoints'], 
                'axisSize':obj_jj['axisSize'], 'rotation':obj_jj['rotation']
            })

        out_obj_info.update({region_idnum:tmp})

    return out_obj_info   

def save_obj_mesh(obj, mesh, source_path):
    #print("name:", (obj['region_name'], obj['obj_name']))
    eightPoints = obj['eightPoints']
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(eightPoints))
    mesh_crop = mesh.crop(aabb)

    o3d.io.write_triangle_mesh(source_path, mesh_crop)

    return len(mesh_crop.vertices)



def main(args, build_name):
    # remove stair
    with open(args.path_category_for_remove_stairs, 'r', encoding='utf-8') as f:   
        align_class = json.load(f)
    remove_stairs = {}
    # for ik, iv in align_class.items():
        # for scene_id, room_map in iv.items():
    stairs_indices = [int(k) for k, v in align_class[build_name].items() if v == "stairs"]
    remove_stairs[build_name] = stairs_indices


    # Load invalid regions that should be skipped
    with open(args.path_invalid_region_id, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    invalid_region = ast.literal_eval(lines[0])
    name_dict = {}
    for out in invalid_region:
        name_dict.update(out)


    df = pd.read_csv(args.path_cat_csv, sep='\t', header=None)  # load category mapping file 
    cat_name_list = df[2] 

    with open(args.path_all_house, 'r') as f:  #load building information
        lines = f.readlines()
    region_obj_info = read_from_house(args, lines, cat_name_list)

    regions_ply = glob.glob(os.path.join(args.path_region_ply, '*.ply')) #  Load all *.ply files 

    # build_level = {}
    region_level = []
  
    for region_path in regions_ply: 
        
        mesh = o3d.io.read_triangle_mesh(region_path)
        region_id = int(region_path.split('/')[-1].replace('region', '').replace('.ply', ''))  
        if region_id in name_dict[build_name]: # Skip invalid regions
            continue
        if region_id in remove_stairs[build_name]:  # Skip/remove stair
            continue
        if region_id in region_obj_info:
            # print(region_id)
            region = region_obj_info[region_id] 


            #save floor mask
            save_path_floor_mask = os.path.join(args.path_output, 'floor_mask')
            os.makedirs(save_path_floor_mask, exist_ok=True)

            mesh = o3d.io.read_triangle_mesh(region_path)
            image = screenshot(mesh)
            last_save_path = os.path.join(save_path_floor_mask, f'region{region_id}.jpg')
            o3d.io.write_image(last_save_path, image)
            make_mask(last_save_path)


            cur_region_path = os.path.join(args.path_output, f'region{region_id}')
            if not os.path.exists(cur_region_path):
                os.makedirs(cur_region_path)
            
           
            source_path_region = os.path.join(cur_region_path, f'region{region_id}.ply')
            o3d.io.write_triangle_mesh(source_path_region, mesh)
            level1 = {'uid': str(uuid.uuid4()), 'name': region[0]['region_name'], 'description':'None', 
                    'category':region[0]['region_name'].split('-')[0], 'categoryId': region[0]['region_name_id'], 'source': source_path_region}
            
            # save building.ply
            source_path_building1 = args.path_all_house.replace('.house', '.ply')
            source_path_building2 = os.path.join(args.path_output, os.path.basename(source_path_building1))
            shutil.copy(source_path_building1, source_path_building2)


            level2 = []
            for i, obj in enumerate(region):
                source_path = os.path.join(cur_region_path, f'{obj["obj_name"]}.ply')

               
                #void_obj_mesh_num = save_obj_mesh(obj, mesh, source_path)
                # if void_obj_mesh_num == 0:
                #     void_obj_mesh_nums = void_obj_mesh_nums + 1

                bbox = {'uid': str(uuid.uuid4()), 'name': obj['obj_name'], 'description':'None', 'category': obj['obj_name'].split('-')[0], 
                        'categoryId': obj['obj_cat_id'], 'parent':'None', 'children':[None], 'relationships':[None], 'source': source_path,
                        'obb':{'centroid': obj['centroid'].tolist(), 'axesLengths': obj['axesLengths'].tolist(), 'normalizedAxes': obj['axes'].tolist(), 
                        'eightPoints': obj['eightPoints'].tolist(), 'prim':[None], 'symmetry':None, 'axisSize':obj['axisSize'], 'rotation':obj['rotation']}
            }
                level2.append(bbox)
            
            #level3 = {'bbox': level2, 'relationships':[None], 'floor_mask': last_save_path}
            level3 = {'bbox': level2, 'relationships':[None], 'floor_mask': [None]}

            # region_level.append(level1 | level3)
            region_level.append({**level1, **level3})
    
    build_level = {'uid': str(uuid.uuid4()), 'level': 'building', 'description': 'None', 'building_level': None, 'source':source_path_building2 ,'scene': region_level}
    
    # output json
    if not os.path.exists(args.path_output_json):
        os.makedirs(args.path_output_json)
    with open(os.path.join(args.path_output_json, 'building_level.json'), "w") as f:
        json.dump(build_level, f, indent=4) 

    print('end!')

if __name__=='__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_region_ply', type=str, default="./data_matterport3d/29hnd4uzFmX/region_segmentations/29hnd4uzFmX/region_segmentations") # every region *.ply
    parser.add_argument('--path_all_house', type=str, default="./data_matterport3d/29hnd4uzFmX/house_segmentations/29hnd4uzFmX/house_segmentations/29hnd4uzFmX.house") # every building information
    parser.add_argument('--path_cat_csv', type=str, default="./data_matterport3d/category_mapping_new.tsv")  # category mapping file
    parser.add_argument('--path_invalid_region_id', type=str, default="./data_matterport3d/old_invalid_region_id.txt")  # Cases in the raw data with scanning failures or empty rooms
    parser.add_argument('--path_category_for_remove_stairs', type=str, default="./data_matterport3d/everyregion_align_category.json") # for remove stairs
    parser.add_argument('--path_output', type=str, default="./data_matterport3d_1/29hnd4uzFmX/")
    parser.add_argument('--path_output_json', type=str, default="./data_matterport3d_1/29hnd4uzFmX/")
    
     
    args = parser.parse_args()

    matport_names1 = ['1LXtFkjw3qL']  # '17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r'
    void_obj_mesh_nums = []
    replace = '29hnd4uzFmX' 
    for build_name in matport_names:
        print('building_name:', build_name)
        args.path_region_ply = args.path_region_ply.replace(replace, build_name)
        args.path_all_house = args.path_all_house.replace(replace, build_name)
        args.path_output = args.path_output.replace(replace, build_name)
        args.path_output_json = args.path_output_json.replace(replace, build_name)

        replace = build_name
         
        main(args, build_name)

    
    
   






