import objaverse
from pathlib import Path
from legent.environment.env_utils import get_default_env_data_path

WALL_MATERIALS = [
    "#A3ABC3", 
    "#AB9E90", 
    "E0DFE3",
    "WorldMaterialsFree_AgedDarkWood",
    "WorldMaterialsFree_BasketWeaveBricks",
    "WorldMaterialsFree_BathroomTiles",
    "WorldMaterialsFree_BrushedIron",
    "WorldMaterialsFree_ClumpMud",
    "WorldMaterialsFree_CoarseConcrete",
    "WorldMaterialsFree_DesertCliffRock",
    "WorldMaterialsFree_DesertSandBrick",
    "WorldMaterialsFree_DryRockyDirt",
    "WorldMaterialsFree_GrassClumps",
    "WorldMaterialsFree_GrassGravel",
    "WorldMaterialsFree_HexBricks",
    "WorldMaterialsFree_PebbledGravel",
    "WorldMaterialsFree_PlainWhiteFabric",
    "WorldMaterialsFree_RuinStoneBricks",
    "WorldMaterialsFree_WavySand",
    "WorldMaterialsFree_WoodenFlooring",
]

FLOOR_MATERIALS = [
    'WorldMaterialsFree_HexBricks', 
    'WorldMaterialsFree_SimpleRedBricks', 
    'WorldMaterialsFree_BathroomTiles', 
    'WorldMaterialsFree_DryRockyDirt', 
    'WorldMaterialsFree_ClumpMud'
]

env_path = Path(get_default_env_data_path())
objaverse_path = env_path / 'filtered_holodeck_objects.json'

def get_objaverse_object(uid):
    objects = objaverse.load_objects([uid])
    return list(objects.values())[0]

if __name__ == '__main__':
    import json
    from tqdm import tqdm
    with open(objaverse_path, 'r') as f:
        objaverse_data = json.load(f)
    windows = [item for item in objaverse_data if item['category'] == 'window']
    for item in tqdm(windows):
        print(get_objaverse_object(item['uid']))
