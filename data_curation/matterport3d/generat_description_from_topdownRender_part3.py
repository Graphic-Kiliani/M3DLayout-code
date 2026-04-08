# import bpy
import os
import json
import math
import openai
import heapq
import numpy as np 
import base64
import glob
from openai import OpenAI
import re
import time
from tqdm import tqdm
import ast
import shutil

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

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
                'ZMojNkEp431', 'zsNo4HB9uLZ'
                ]

def generate_and_save_descriptions_from_dataset(
    api_key= None,
    dataset_dir= None,
    save_description_path= None,
    cur_region_category = None,
    name= None,
    which_view = None,
    desript = None,
    model_name= "gpt-4o",
    max_tokens= 800
):
    client = OpenAI(api_key=api_key, timeout=30)
    # We provide three types of prompts:
    # - prompt_text: used to obtain the ground-truth description of the room
    # - prompt_text_style: used to obtain the room's style, type, and shape/size description
    # - prompt_mask: used to obtain the room's shape and size description
    # Choose the appropriate prompt as needed.
    # In practice, we use these three prompts sequentially to obtain all fields of the room layout description.
    
    prompt_text = (
        f"Analyze the specific room ({cur_region_category}) composed of a series of bounding boxes in English. "  
        "You are an interior layout analyst. Analyze the provided top-down view of the room, "
        "and a top-down view of major objects potentially containing small objects, and output a structured JSON description. "
        "Your task is to describe the room in detail for training a 2D-to-3D roomgeneration model. "
        "Strive to include descriptions for as many of the provided, text-annotated objects as possible. "
        "Base your analysis strictly on the visual content, Avoid any uncertain orspeculative expressions like 'appears to be', 'might be','likely', or 'a possible'. "
        "Only describe what is clearly visible, and express all content as completeEnglish sentences. Do not use single words or fragments. "
        "Note that these descriptions can not contain keywords related to bounding boxes color(eg. orange, red, hub, tone, colorful and so on) and 'bounding boxes'. "
        "Your output must follow the exact JSON format below, where each value is a listof complete sentences in English, If a section is not applicable, use an empty list. "
        "Describe all section as much as possible. "
        "The categories are as follows:\n\n"
        "{\n"
        '    "global_description": {\n'
        '    "room_type": [],\n'
        '    "style": [],\n'
        '   "shape_and_size": [],\n'
        '    "local_density": [],\n'
        '    "symmetry": [],\n'
        '    "functional_zones": []\n'
        '    },\n'
        '    "large_objects_description": {\n'
        '            "summary": [],\n'
        '            "abosolute_arrangement": [],\n'
        '            "relative_positions": []\n'
        '            },\n'
        '    "small_objects_description": {\n'
        '            "summary": [],\n'
        '            "on_surfaces": [],\n'
        '            "on_shelves": [],\n'
        '            "distribution_and_pattern": []\n'
        '    }\n'
        "}\n\n"
            "Do not include any explanation or extra text, Only output the JSON object asdescribed above, and use '\n' for line breaks. "
            #f"Here is an output example: {example_data}"
        )

    prompt_text_style = (
        f"Analyze the provided spherical panorama image of a specific room ({cur_region_category}) in English. "
        "Your task is to describe the room style, room type and shape or size in detail for training a 2D-to-3D roomgeneration model. "
        "The <room style> must be diversified for better training effect, choosing from styles such as 'modern', 'minimalist', 'scandinavian', 'traditional', 'industrial', 'Farmhouse', 'tramsitional', 'rustic', 'elegant' or 'eclectic'. "
        f"The <room type> must be include the room name {cur_region_category}. "
        "Do not invent styles which are not visually supported. "
        "Your output must follow the exact JSON format below, where each value is a listof complete sentences in English. "
        # "And do not include any explanation or extra text unrelated to the current room, Only output the JSON object asdescribed above, and use '\n' for line breaks. "
        "The categories are as follows:\n\n"
        "{\n"
        '    "global_description": {\n'
        '    "room_type": [],\n'
        '    "style": [],\n'
        '   "shape_and_size": [],\n'
        '    }\n'
        "}\n\n"
        "And do not include any explanation or extra text unrelated to the current room, Only output the JSON object asdescribed above. "
        "Please output only a valid JSON string without any markdown formatting or code block markers. "
        )
# as concise as possible
    prompt_mask = (
        "You are a house type analyst and good at explaining interior space structure and layout. Analyze the provided masked floor plan of a room in English. "
        "Your task is to describe the shape and size of floor plan for training a 2D-to-3D room generation model. "
        "Your output must follow the exact JSON format below, where each value is a listof complete sentences in English. "
        "Do not include any descriptions not related to the <shape and size>. "
        "Don't summarize. Avoid excessive reasoning and imagination. "
        #"Keep the description as concise as possible. "
        "The categories are as follows:\n\n"
        "{\n"
        '   "shape_and_size": [],\n'
        "}\n\n"
        # "And do not include any explanation or extra text unrelated to the current room, Only output the JSON object asdescribed above. "
        "Please output only a valid JSON string without any markdown formatting or code block markers. "
        )

    out = None
    try:
        with open(dataset_dir, "rb") as image_file:
            base64_image1 = base64.b64encode(image_file.read()).decode('utf-8')
        # with open(dataset_dir[1], "rb") as image_file:
        #     base64_image2 = base64.b64encode(image_file.read()).decode('utf-8')

        response1 = client.chat.completions.create(
            model=model_name,
            messages=[
                #{"role": "system", "content": f"You are a house type analyst and good at explaining interior space structure and layout. "},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image1}"}},
                        {"type": "text", "text": prompt_mask},
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        description_text1 = response1.choices[0].message.content.strip()   


        if 'iamge' in description_text1:
            description_text1.replace('iamge', 'room')

        with open(save_description_path, "w", encoding="utf-8") as json_file:
            json.dump({"description": description_text1}, json_file, indent=2, ensure_ascii=False)
        
    
    except Exception as e:
        print(f"[Error]: {e}")
        print(f"[Error]:{(name, dataset_dir)}")
        out = e
    
    # time.sleep(1) 
    return out, name, dataset_dir




if __name__ == '__main__':
    
    os.environ['OPENAI_API_KEY'] = ''   # add API key
    
    path_root = './3Dbbox_top_down_image/17DRP5sb8fy/'


    replace = '17DRP5sb8fy'      

    dict_false = {}
    for i, name in enumerate(matport_names):
        print("building name:", (i+1, name))

        path_root = path_root.replace(replace, name)

        replace = name

        all_folder = [name for name in os.listdir(path_root) if os.path.isdir(os.path.join(path_root, name))]
        
        all_region_image = os.listdir(path_root)
        for single_region_image in all_region_image:
            top_down_bbx_image = os.path.join(path_root, single_region_image)
           
            description_json_path = top_down_bbx_image.replace('png', 'json')
 
            cur_region_category = single_region_image.split("_")[0]


            e, Rname, Rdataset_dir = generate_and_save_descriptions_from_dataset(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        dataset_dir=top_down_bbx_image,
                        save_description_path=description_json_path,
                        cur_region_category=cur_region_category,
                        name=name
                    )
            
            if e is not None:
                if Rname not in dict_false:
                    dict_false[Rname] = [] 
                dict_false[Rname].append(Rdataset_dir)
    
    with open("./descr_hierar_all/false_topdownImage_path.txt", "w") as f:
        f.write(str(dict_false))


