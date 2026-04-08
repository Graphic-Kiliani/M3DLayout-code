## Matterport3D Processing Pipeline

We provide a complete processing pipeline for the Matterport3D dataset.

Before running the pipeline, please download the Matterport3D data from the official repository:

👉 https://github.com/niessner/Matterport/tree/master

After downloading, place the data under `./data_matterport3d`.

## (0) File Description

- `new_invalid_region_id.txt` and `old_invalid_region_id.txt`  
  Cases in the raw data with scanning failures or empty rooms.

- `everyregion_align_category.json`  
  Preprocessed correspondence between regions and categories. For example:

"17DRP5sb8fy": {
  "9": "kitchen",
  "2": "toilet",
  "6": "entryway/foyer/lobby",
  "3": "bathroom",
  "5": "dining room",
  "0": "bedroom",
  "8": "familyroom",
  "4": "tv (must have theater style seating)",
  "1": "bathroom",
  "7": "living room"
}

Here, "9" denotes the 9th region/room in the current building 17DRP5sb8fy, and "kitchen" indicates that the category of region 9 is kitchen.


## (1) Process the Matterport3D Raw Data

Process the raw Matterport3D data to obtain the 3D bounding box information of each room, floor masks, individual object meshes, etc.

`python preprocessing_part1.py`

## (2) Generate Top-Down Views of Rooms Represented by 3D Bounding Boxes

Generate top-down views of rooms represented by 3D bounding boxes, which are used in the next step to generate room descriptions.

`python 3Dbbox_top_down_part2.py`

## (3) Generate Rome Descriptions

Generate rome text descriptions based on the top-down views obtained in Step (2).

⚠️ Note:

The top-down views obtained in Step (2) do not include the room style or floor shape information. Therefore, we additionally use panoramic images (provided by the Matterport3D raw data) and the floor plan masks obtained in Step (1) to generate the "style" and "shape" fields in the rome descriptions.

`python generat_description_from_topdownRender_part3.py`

## (4) Merge Descriptions into Final Data

By filling the fields generated in Step (3) into the corresponding entries of the JSON file obtained in Step (1), the final processed data can be obtained.

