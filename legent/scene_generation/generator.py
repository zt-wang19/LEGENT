import copy
import json
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import Polygon

from legent.scene_generation.doors import default_add_doors
from legent.scene_generation.house import generate_house_structure
from legent.scene_generation.objects import ObjectDB
from legent.scene_generation.room import Room
from legent.scene_generation.room_spec import RoomSpec
from legent.scene_generation.small_objects import add_small_objects
from legent.server.rect_placer import RectPlacer
from legent.utils.io import log
from legent.utils.math import look_rotation
from legent.scene_generation.utils import (
    get_objaverse_object,
    objaverse_path,
    WALL_MATERIALS,
    FLOOR_MATERIALS,
)

from .asset_groups import Asset
from .constants import (
    MARGIN,
    P_CHOOSE_ASSET_GROUP,
    P_W1_ASSET_SKIPPED,
    PADDING_AGAINST_WALL,
    # UNIT_SIZE,
)
from .types import Vector3

DEFAULT_FLOOR_SIZE = 2.5
# HALF_UNIT_SIZE = UNIT_SIZE / 2  # Half of the size of a unit in the grid
# SCALE_RATIO = UNIT_SIZE / DEFAULT_FLOOR_SIZE
DEFAULT_WALL_PREFAB = "LowPolyInterior2_Wall1_C1_01"
MAX_SPECIFIED_RECTANGLE_RETRIES = 10
MAX_SPECIFIED_NUMBER = 20
WALL_THICKNESS = 0.075


# def log(*args, **kwargs):
#     pass


class HouseGenerator:
    def __init__(
        self,
        room_spec: Optional[Union[RoomSpec, str]] = None,
        dims: Optional[Tuple[int, int]] = None,
        objectDB: ObjectDB = None,
        unit_size=2.5,
    ) -> None:
        self.room_spec = room_spec
        self.dims = dims
        self.odb = objectDB
        self.rooms: Dict[str, Room] = dict()
        self.unit_size = unit_size
        self.half_unit_size = unit_size / 2  # Half of the size of a unit in the grid
        self.scale_ratio = unit_size / DEFAULT_FLOOR_SIZE
        self.wall_height = 3

    def generate_structure(self, room_spec):
        house_structure = generate_house_structure(
            room_spec=room_spec, dims=self.dims, unit_size=self.unit_size
        )
        return house_structure

    def format_object(self, prefab, position, rotation, scale):

        object = {
            "prefab": prefab,
            "position": [position[0], position[1], position[2]],
            "rotation": [0, rotation, 0],
            "scale": scale,
            "type": "kinematic",
        }
        return object

    def add_floors_and_walls(
        self,
        house_structure,
        room_spec,
        odb,
        prefabs,
    ):
        res = {}

        room_num = len(room_spec.room_type_map)
        room_ids = set(room_spec.room_type_map.keys())

        # floor_materials = ["#FFFFFF"]
        # floor_material = random.choice(FLOOR_MATERIALS)
        # wall_material = random.choice(WALL_MATERIALS)
        floor_material = "#FFFFFF"
        wall_material = "#FFFFFF"

        door_prefab = "LowPolyInterior2_Door1_C1"
        door_x_size = prefabs[door_prefab]["size"]["x"]
        door_y_size = prefabs[door_prefab]["size"]["y"]
        door_z_size = prefabs[door_prefab]["size"]["z"]

        floor_thickness = 0.2
        wall_thickness = 0.05
        wall_height = self.wall_height

        floors = house_structure.floorplan
        # convert 1 in floors to 0
        floors = np.where(floors == 1, 0, floors)
        log(f"floors:\n{floors}")
        # exit()

        doors = default_add_doors(odb, room_spec, house_structure)
        log(f"doors: {doors}")
        door_positions = set(doors.values())
        floor_positions = []
        for i in range(floors.shape[0]):
            for j in range(floors.shape[1]):
                if floors[i][j] != 0:
                    floor_positions.append((i, j))
        min_floor_x = min([x for x, y in floor_positions])
        max_floor_x = max([x for x, y in floor_positions])
        min_floor_z = min([y for x, y in floor_positions])
        max_floor_z = max([y for x, y in floor_positions])

        min_floor_x_pos = (min_floor_x - 1) * self.unit_size
        max_floor_x_pos = (max_floor_x) * self.unit_size
        min_floor_z_pos = (min_floor_z - 1) * self.unit_size
        max_floor_z_pos = (max_floor_z) * self.unit_size

        floor_center_x = (min_floor_x_pos + max_floor_x_pos) / 2
        floor_center_z = (min_floor_z_pos + max_floor_z_pos) / 2

        floor_x_size = max_floor_x_pos - min_floor_x_pos
        floor_z_size = max_floor_z_pos - min_floor_z_pos
        res["floors"] = [
            {
                "position": [floor_center_x, -floor_thickness / 2, floor_center_z],
                "size": [floor_x_size, floor_thickness, floor_z_size],
                "rotation": [0, 0, 0],
                "material": floor_material,
            },
        ]
        add_ceiling = True
        if add_ceiling:
            res['floors'].append({
                "position": [
                    floor_center_x,
                    wall_height + floor_thickness / 2,
                    floor_center_z,
                ],
                "size": [floor_x_size, floor_thickness, floor_z_size],
                "rotation": [180, 0, 0],
                "material": floor_material,
            })

        walls = []
        door_holes = []
        for i in range(floors.shape[0] - 1):
            for j in range(floors.shape[1] - 1):
                # log(f'{i} {j}')
                # log(f'{floors[i][j]} {floors[i+1][j]} {floors[i][j+1]} {floors[i+1][j+1]}')

                if i + 1 < floors.shape[0] and floors[i][j] != floors[i + 1][j]:

                    has_door = True if ((i, j), (i + 1, j)) in door_positions else False
                    begin_point_x = i * self.unit_size
                    begin_point_z = (j - 1) * self.unit_size
                    end_point_x = i * self.unit_size
                    end_point_z = j * self.unit_size

                    out = True if floors[i][j] * floors[i + 1][j] == 0 else False

                    wall = {
                        "begin": (begin_point_x, begin_point_z),
                        "end": (end_point_x, end_point_z),
                        "direction": 0,
                        "has_door": has_door,
                        "out": out,
                    }
                    walls.append(wall)
                    if ((i, j), (i + 1, j)) in door_positions:
                        door_holes.append(wall)
                if j + 1 < floors.shape[1] and floors[i][j] != floors[i][j + 1]:

                    has_door = True if ((i, j), (i, j + 1)) in door_positions else False
                    begin_point_x = (i - 1) * self.unit_size
                    begin_point_z = j * self.unit_size
                    end_point_x = i * self.unit_size
                    end_point_z = j * self.unit_size

                    out = True if floors[i][j] * floors[i + 1][j] == 0 else False

                    wall = {
                        "begin": (begin_point_x, begin_point_z),
                        "end": (end_point_x, end_point_z),
                        "direction": 1,
                        "has_door": has_door,
                        "out": out,
                    }
                    walls.append(wall)
                    if ((i, j), (i, j + 1)) in door_positions:
                        door_holes.append(wall)
        # for wall in walls:
        #     log(wall)
        merge_walls = []
        for wall in walls:
            # log(merge_walls)
            if len(merge_walls) == 0:
                merge_walls.append(wall)
            else:
                merge_flag = False
                for merge_wall in merge_walls:
                    if (
                        merge_wall["begin"] == wall["end"]
                        and merge_wall["direction"] == wall["direction"]
                    ):
                        conflict = False
                        for other_wall in walls:
                            if (
                                other_wall["begin"] == merge_wall["begin"]
                                or other_wall["end"] == merge_wall["begin"]
                            ) and (other_wall["direction"] != merge_wall["direction"]):
                                conflict = True
                                break
                        if not conflict:
                            merge_wall["begin"] = wall["begin"]
                            merge_wall["has_door"] |= wall["has_door"]
                            merge_wall["out"] |= wall["out"]
                            merge_flag = True

                    elif (
                        merge_wall["end"] == wall["begin"]
                        and merge_wall["direction"] == wall["direction"]
                    ):
                        conflict = False
                        for other_wall in walls:
                            if (
                                other_wall["begin"] == merge_wall["end"]
                                or other_wall["end"] == merge_wall["end"]
                            ) and (other_wall["direction"] != merge_wall["direction"]):
                                conflict = True
                                break
                        if not conflict:
                            merge_wall["end"] = wall["end"]
                            merge_wall["has_door"] |= wall["has_door"]
                            merge_wall["out"] |= wall["out"]
                            merge_flag = True
                if not merge_flag:
                    merge_walls.append(wall)

        def sample_door_position(begin, end, door_size):
            begin += door_size / 2
            end -= door_size / 2
            if begin > end:
                return None
            return random.uniform(begin, end)
        
        def filter_window_size(asset):
            # return True
            if asset['size']['x'] > 1.5:
                return False
            if asset['size']['y'] > 2:
                return False
            if asset['size']['z'] > 0.5:
                return False
            return True

        objaverse_assets = json.load(open(objaverse_path, "r"))
        windows = [asset for asset in objaverse_assets if asset["category"] == "window" and filter_window_size(asset)]
        window_asset = random.choice(windows)
        window_asset = [item for item in windows if item['uid']=='e00ee30713dd4b63a16aaceee3dc8585'][0]
        window_uid = window_asset["uid"]
        window_prefab = get_objaverse_object(window_uid)
        log(f"window_prefab: {window_prefab}")
        log(f'window_asset["size"]: {window_asset["size"]}')
        reverse_rotation = (
            True if window_asset["size"]["x"] < window_asset["size"]["z"] else False
        )
        # reverse_rotation = not reverse_rotation

        window_x_size = window_asset["size"]["x"]
        window_y_size = window_asset["size"]["y"]
        window_z_size = window_asset["size"]["z"]
        window_x_scale = 1
        window_y_scale = 1
        window_z_scale = 1
        # window_x_size = 0.5
        # window_y_size = 1
        # window_z_size = wall_thickness
        # window_x_scale = window_x_size / window_asset["size"]["x"]
        # window_y_scale = window_y_size / window_asset["size"]["y"]
        # window_z_scale = window_z_size / window_asset["size"]["z"]
        

        WALL_OBJECT_MAX_RETRIES = 10
        doors = []
        windows = []

        def hole_intersect(hole1, hole2):
            x1, y1 = hole1["position_xy"]
            x2, y2 = hole2["position_xy"]
            w1, h1 = hole1["size_xy"]
            w2, h2 = hole2["size_xy"]
            if x1 + w1 / 2 < x2 - w2 / 2 or x2 + w2 / 2 < x1 - w1 / 2:
                return False
            if y1 + h1 / 2 < y2 - h2 / 2 or y2 + h2 / 2 < y1 - h1 / 2:
                return False
            return True

        def can_place_hole(wall, hole):
            for h in wall["holes"]:
                if hole_intersect(h, hole):
                    return False
            return True

        for wall in merge_walls:
            wall["holes"] = []
            begin_point_x, begin_point_z = wall["begin"]
            end_point_x, end_point_z = wall["end"]

            wall_center_x = (begin_point_x + end_point_x) / 2
            wall_center_y = wall_height / 2
            wall_center_z = (begin_point_z + end_point_z) / 2

            wall["size"] = {
                "x": (
                    end_point_z - begin_point_z
                    if wall["direction"] == 0
                    else end_point_x - begin_point_x
                ),
                "y": wall_height,
            }

            if wall["has_door"]:
                direction = wall["direction"]
                if direction == 0:

                    door_z = sample_door_position(
                        begin_point_z, end_point_z, door_x_size
                    )
                    door = {
                        "position": [begin_point_x, door_y_size / 2, door_z],
                        "rotation": [0, 90, 0],
                        "scale": [1, 1, 1],
                        "type": "kinematic",
                        "prefab": door_prefab,
                    }
                    hole = {
                        "position_xy": [
                            door_z - wall_center_z,
                            door_y_size / 2 - wall_center_y,
                        ],
                        "size_xy": [door_x_size, door_y_size],
                    }
                    # wall["holes"] = [hole]
                    door_bbox = (
                        door["position"][0] - door_z_size / 2 - 1,
                        door["position"][2] - door_x_size / 2,
                        door["position"][0] + door_z_size / 2 + 1,
                        door["position"][2] + door_x_size / 2,
                    )
                else:
                    door_x = sample_door_position(
                        begin_point_x, end_point_x, door_x_size
                    )
                    door = {
                        "position": [door_x, door_y_size / 2, begin_point_z],
                        "rotation": [0, 0, 0],
                        "scale": [1, 1, 1],
                        "type": "kinematic",
                        "prefab": door_prefab,
                    }
                    hole = {
                        "position_xy": [
                            door_x - wall_center_x,
                            door_y_size / 2 - wall_center_y,
                        ],
                        "size_xy": [door_x_size, door_y_size],
                    }
                    # wall["holes"] = [hole]
                    door_bbox = (
                        door["position"][0] - door_x_size / 2,
                        door["position"][2] - door_z_size / 2 - 1,
                        door["position"][0] + door_x_size / 2,
                        door["position"][2] + door_z_size / 2 + 1,
                    )

                self.placer.insert("door", door_bbox)
                doors.append(door)
                wall["holes"] = [hole]
            if wall["out"]:
                placed_hole = None
                for _ in range(WALL_OBJECT_MAX_RETRIES):
                    x = random.uniform(
                        0 + window_x_size / 2, wall["size"]["x"] - window_x_size / 2
                    )
                    y = random.uniform(
                        1 + window_x_size / 2, wall["size"]["y"] - window_y_size / 2
                    )  # 1 for window minimum height
                    hole = {
                        "position_xy": [
                            x - wall["size"]["x"] / 2,
                            y - wall["size"]["y"] / 2,
                        ],
                        "size_xy": [window_x_size, window_y_size],
                    }
                    if can_place_hole(wall, hole):
                        wall["holes"].append(hole)
                        placed_hole = hole
                        break
                if placed_hole:
                    window_x_pos = wall_center_x + (
                        placed_hole["position_xy"][0] if wall["direction"] == 1 else 0
                    )
                    window_y_pos = wall_center_y + placed_hole["position_xy"][1]
                    window_z_pos = wall_center_z + (
                        placed_hole["position_xy"][0] if wall["direction"] == 0 else 0
                    )
                    window = {
                        "position": [window_x_pos, window_y_pos, window_z_pos],
                        "rotation": [
                            0,
                            (
                                90
                                if (wall["direction"] == 0 and not reverse_rotation)
                                or (wall["direction"] == 1 and reverse_rotation)
                                else 0
                            ),
                            0,
                        ],
                        "scale": [window_x_scale, window_y_scale, window_z_scale],
                        "type": "kinematic",
                        "prefab": window_prefab,
                    }
                    windows.append(window)

        res["walls"] = []
        for wall in merge_walls:
            begin_point_x, begin_point_z = wall["begin"]
            end_point_x, end_point_z = wall["end"]
            direction = wall["direction"]
            if direction == 0:

                wall_x_size = end_point_z - begin_point_z
                wall_z_size = wall_thickness
                rotation = [0, 270, 0]
            else:
                wall_x_size = end_point_x - begin_point_x
                wall_z_size = wall_thickness
                rotation = [0, 0, 0]
            wall_center_x = (begin_point_x + end_point_x) / 2
            wall_center_z = (begin_point_z + end_point_z) / 2
            res["walls"].append(
                {
                    "position": [wall_center_x, wall_height / 2, wall_center_z],
                    "size": [wall_x_size, wall_height, wall_z_size],
                    "rotation": rotation,
                    "material": wall_material,
                    "holes": wall.get("holes", []),
                    "has_door": wall.get("has_door", False),
                }
            )

        return res, doors, windows

    def add_human_and_agent(self, floors):
        def get_bbox_of_floor(x, z):
            x, z = (x - 0.5) * self.unit_size, (z - 0.5) * self.unit_size
            return (
                x - self.half_unit_size,
                z - self.half_unit_size,
                x + self.half_unit_size,
                z + self.half_unit_size,
            )

        def random_xz_for_agent(
            eps, floors
        ):  # To prevent being positioned in the wall and getting pushed out by collision detection.
            # ravel the floor
            ravel_floors = floors.ravel()
            # get the index of the floor
            floor_idx = np.where(ravel_floors != 0)[0]
            # sample from the floor index
            floor_idx = np.random.choice(floor_idx)
            # get the x and z index
            x, z = np.unravel_index(floor_idx, floors.shape)
            log(f"human/agent x: {x}, z: {z}")

            # get the bbox of the floor
            bbox = get_bbox_of_floor(x, z)
            # uniformly sample from the bbox, with eps
            x, z = np.random.uniform(bbox[0] + eps, bbox[2] - eps), np.random.uniform(
                bbox[1] + eps, bbox[3] - eps
            )
            return x, z

        ### STEP 3: Randomly place the player and playmate (AI agent)
        # place the player
        AGENT_HUMAN_SIZE = 1
        while True:
            x, z = random_xz_for_agent(eps=0.5, floors=floors)
            player = {
                "prefab": "",
                "position": [x, 0.05, z],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": [1, 1, 1],
                "parent": -1,
                "type": "",
            }
            ok = self.placer.place("playmate", x, z, AGENT_HUMAN_SIZE, AGENT_HUMAN_SIZE)

            if ok:
                log(f"player x: {x}, z: {z}")
                break
        # place the playmate
        while True:
            x, z = random_xz_for_agent(eps=0.5, floors=floors)
            playmate = {
                "prefab": "",
                "position": [x, 0.05, z],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": [1, 1, 1],
                "parent": -1,
                "type": "",
            }
            ok = self.placer.place("playmate", x, z, AGENT_HUMAN_SIZE, AGENT_HUMAN_SIZE)
            if ok:
                log(f"playmate x: {x}, z: {z}")
                break

        # player lookat the playmate
        vs, vt = np.array(player["position"]), np.array(playmate["position"])
        vr = look_rotation(vt - vs)
        player["rotation"] = [0, vr[1], 0]

        return player, playmate

    def get_floor_polygons(self, xz_poly_map: dict) -> Dict[str, Polygon]:
        """Return a shapely Polygon for each floor in the room."""
        floor_polygons = dict()
        for room_id, xz_poly in xz_poly_map.items():
            floor_polygon = []
            for (x0, z0), (x1, z1) in xz_poly:
                floor_polygon.append((x0, z0))
            floor_polygon.append((x1, z1))
            floor_polygons[f"room|{room_id}"] = Polygon(floor_polygon)
        return floor_polygons

    def get_rooms(self, room_type_map, floor_polygons):
        for room_id, room_type in room_type_map.items():
            polygon = floor_polygons[f"room|{room_id}"]
            room = Room(
                polygon=polygon,
                room_type=room_type,
                room_id=room_id,
                odb=self.odb,
            )
            self.rooms[room_id] = room

    def sample_and_add_floor_asset(
        self,
        room: Room,
        rectangle: Tuple[float, float, float, float],
        anchor_type: str,
        anchor_delta: int,
        odb: ObjectDB,
        spawnable_assets,  # pd.DataFrame
        spawnable_asset_groups,  # pd.DataFrame
        priority_asset_types: List[str],
    ):
        set_rotated = None

        # NOTE: Choose the valid rotations
        x0, z0, x1, z1 = rectangle
        rect_x_length = x1 - x0
        rect_z_length = z1 - z0

        # NOTE: add margin to each object.
        # NOTE: z is the forward direction on each object.
        # Therefore, we only add space in front of the object.
        if anchor_type == "onEdge":
            x_margin = 2 * MARGIN["edge"]["sides"]
            z_margin = (
                MARGIN["edge"]["front"] + MARGIN["edge"]["back"] + PADDING_AGAINST_WALL
            )
        elif anchor_type == "inCorner":
            x_margin = 2 * MARGIN["corner"]["sides"] + PADDING_AGAINST_WALL
            z_margin = (
                MARGIN["corner"]["front"]
                + MARGIN["corner"]["back"]
                + PADDING_AGAINST_WALL
            )
        elif anchor_type == "inMiddle":
            # NOTE: add space to both sides
            x_margin = 2 * MARGIN["middle"]
            z_margin = 2 * MARGIN["middle"]

        # NOTE: define the size filters
        if anchor_delta in {1, 7}:
            # NOTE: should not be rotated
            size_filter = lambda assets_df: (
                (assets_df["xSize"] + x_margin < rect_x_length)
                & (assets_df["zSize"] + z_margin < rect_z_length)
            )
            set_rotated = False
        elif anchor_delta in {3, 5}:
            # NOTE: must be rotated
            size_filter = lambda assets_df: (
                (assets_df["zSize"] + z_margin < rect_x_length)
                & (assets_df["xSize"] + x_margin < rect_z_length)
            )
            set_rotated = True
        else:
            # NOTE: either rotated or not rotated works
            size_filter = lambda assets_df: (
                (
                    (assets_df["xSize"] + x_margin < rect_x_length)
                    & (assets_df["zSize"] + z_margin < rect_z_length)
                )
                | (
                    (assets_df["zSize"] + z_margin < rect_x_length)
                    & (assets_df["xSize"] + x_margin < rect_z_length)
                )
            )

        asset_group_candidates = spawnable_asset_groups[
            spawnable_asset_groups[anchor_type] & size_filter(spawnable_asset_groups)
        ]
        asset_candidates = spawnable_assets[
            spawnable_assets[anchor_type] & size_filter(spawnable_assets)
        ]

        if priority_asset_types:
            for asset_type in priority_asset_types:
                asset_type = asset_type.lower()
                # NOTE: see if there are any semantic asset groups with the asset
                asset_groups_with_type = asset_group_candidates[
                    asset_group_candidates[f"has{asset_type}"]
                ]

                # NOTE: see if assets can spawn by themselves
                can_spawn_alone_assets = odb.PLACEMENT_ANNOTATIONS[
                    odb.PLACEMENT_ANNOTATIONS.index == asset_type
                ]

                can_spawn_standalone = len(can_spawn_alone_assets) and (
                    can_spawn_alone_assets[f"in{room.room_type}s"].iloc[0] > 0
                )
                assets_with_type = None
                if can_spawn_standalone:
                    assets_with_type = asset_candidates[
                        asset_candidates["assetType"] == asset_type
                    ]

                # NOTE: try using an asset group first
                if len(asset_groups_with_type) and (
                    assets_with_type is None or random.random() <= P_CHOOSE_ASSET_GROUP
                ):
                    # NOTE: Try using an asset group
                    asset_group = asset_groups_with_type.sample()
                    chosen_asset_group = room.place_asset_group(
                        asset_group=asset_group,
                        set_rotated=set_rotated,
                        rect_x_length=rect_x_length,
                        rect_z_length=rect_z_length,
                    )
                    if chosen_asset_group is not None:
                        return chosen_asset_group

                # NOTE: try using a standalone asset
                if assets_with_type is not None and len(assets_with_type):
                    # NOTE: try spawning in standalone
                    asset = assets_with_type.sample()
                    return room.place_asset(
                        asset=asset,
                        set_rotated=set_rotated,
                        rect_x_length=rect_x_length,
                        rect_z_length=rect_z_length,
                    )
        # NOTE: try using an asset group
        can_use_asset_group = True
        must_use_asset_group = False

        if (
            len(asset_group_candidates)
            and random.random() <= P_CHOOSE_ASSET_GROUP
            and can_use_asset_group
        ) or (must_use_asset_group and len(asset_group_candidates)):

            # NOTE: use an asset group if you can
            asset_group = asset_group_candidates.sample()
            chosen_asset_group = room.place_asset_group(
                asset_group=asset_group,
                set_rotated=set_rotated,
                rect_x_length=rect_x_length,
                rect_z_length=rect_z_length,
            )
            if chosen_asset_group is not None:
                return chosen_asset_group
            return chosen_asset_group

        # NOTE: Skip weight 1 assets with a probability of P_W1_ASSET_SKIPPED
        if random.random() <= P_W1_ASSET_SKIPPED:
            asset_candidates = asset_candidates[
                asset_candidates[f"in{room.room_type}s"] != 1
            ]

        # NOTE: no assets fit the anchor_type and size criteria
        if not len(asset_candidates):
            return None

        # NOTE: this is a sampling by asset type
        asset_type = random.choice(asset_candidates["assetType"].unique())
        asset = asset_candidates[asset_candidates["assetType"] == asset_type].sample()
        return room.place_asset(
            asset=asset,
            set_rotated=set_rotated,
            rect_x_length=rect_x_length,
            rect_z_length=rect_z_length,
        )

    def get_spawnable_asset_group_info(self):
        import pandas as pd
        from .asset_groups import AssetGroupGenerator

        asset_groups = self.odb.ASSET_GROUPS

        data = []
        for asset_group_name, asset_group_data in asset_groups.items():
            asset_group_generator = AssetGroupGenerator(
                name=asset_group_name,
                data=asset_group_data,
                odb=self.odb,
            )

            dims = asset_group_generator.dimensions
            group_properties = asset_group_data["groupProperties"]

            # NOTE: This is kinda naive, since a single asset in the asset group
            # could map to multiple different types of asset types (e.g., Both Chair
            # and ArmChair could be in the same asset).
            # NOTE: use the asset_group_generator.data instead of asset_group_data
            # since it only includes assets from a given split.
            asset_types_in_group = set(
                asset_type
                for asset in asset_group_generator.data["assetMetadata"].values()
                for asset_type, asset_id in asset["assetIds"]
            )
            group_data = {
                "assetGroupName": asset_group_name,
                "assetGroupGenerator": asset_group_generator,
                "xSize": dims["x"],
                "ySize": dims["y"],
                "zSize": dims["z"],
                "inBathrooms": group_properties["roomWeights"]["bathrooms"],
                "inBedrooms": group_properties["roomWeights"]["bedrooms"],
                "inKitchens": group_properties["roomWeights"]["kitchens"],
                "inLivingRooms": group_properties["roomWeights"]["livingRooms"],
                "allowDuplicates": group_properties["properties"]["allowDuplicates"],
                "inCorner": group_properties["location"]["corner"],
                "onEdge": group_properties["location"]["edge"],
                "inMiddle": group_properties["location"]["middle"],
            }

            # NOTE: Add which types are in this asset group
            for asset_type in self.odb.OBJECT_DICT.keys():
                group_data[f"has{asset_type}"] = asset_type in asset_types_in_group

            data.append(group_data)

        return pd.DataFrame(data)

    def prefab_fit_rectangle(self, prefab_size, rectangle):
        x0, z0, x1, z1 = rectangle
        rect_x_length = x1 - x0
        rect_z_length = z1 - z0
        prefab_x_length = prefab_size["x"]
        prefab_z_length = prefab_size["z"]
        if (prefab_x_length < rect_x_length * 0.9) and (
            prefab_z_length < rect_z_length * 0.9
        ):
            return 0
        elif (prefab_x_length < rect_z_length * 0.9) and (
            prefab_z_length < rect_x_length * 0.9
        ):
            return 90
        else:
            return -1

    def add_corner_agent(self, max_x, max_z):

        agents = []
        AGENT_SIZE = 0.3

        CORNER_MARGIN = 0.2
        for i, (x, z) in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
            offset_x = 1 if x == 0 else -1
            offset_z = 1 if z == 0 else -1
            x = x * max_x + offset_x * (WALL_THICKNESS + CORNER_MARGIN)
            z = z * max_z + offset_z * (WALL_THICKNESS + CORNER_MARGIN)
            from legent.utils.io import log

            bbox = (
                x - AGENT_SIZE / 2,
                z - AGENT_SIZE / 2,
                x + AGENT_SIZE / 2,
                z + AGENT_SIZE / 2,
            )
            # if self.placer.place("agent", x, z, AGENT_SIZE, AGENT_SIZE):
            if not self.placer.spindex.intersect(bbox):
                # log(i)

                rotation = 45 + i * 90
                # log(rotation)
                agent = {
                    "prefab": "",
                    "position": [x, 0.05, z],
                    "rotation": [0, rotation, 0],
                    "scale": [1, 1, 1],
                    "parent": -1,
                    "type": "",
                }
                agents.append(agent)
        if agents:
            idx = random.randint(0, len(agents) - 1)
            agent = agents[idx]
            x = agent["position"][0]
            z = agent["position"][2]
            bbox = (
                x - AGENT_SIZE / 2,
                z - AGENT_SIZE / 2,
                x + AGENT_SIZE / 2,
                z + AGENT_SIZE / 2,
            )
            # self.placer.insert("agent", bbox)
            return True, agent
        return False, None

    def get_light(self):
        lights = []
        light_instances = []
        for room in self.rooms.values():
            id = room.room_id
            polygon = list(room.room_polygon.polygon.exterior.coords)
            min_x = min([x for x, y in polygon])
            max_x = max([x for x, y in polygon])
            min_z = min([y for x, y in polygon])
            max_z = max([y for x, y in polygon])
            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2
            light_position = [center_x, self.wall_height, center_z]
            light_rotation = [0, 0, 0]
            light = {
                "name": "SpotLight0",
                "lightType": "Spot",  # Point, Spot, Directional
                "position": light_position,
                "rotation": light_rotation,
                "spotAngle": 180.0,
                "useColorTemperature": True,
                "colorTemperature": 5500.0,
                "color": [1.0, 1.0, 1.0],
                "intensity": 15,  # brightness
                "range": 15,
                "shadowType": "None",
            }
            lights.append(light)
            light_instance = {
                "position": light_position,
                "rotation": light_rotation,
                "size": [0.8, 0.01, 0.8],
                "material": "Light",
            }
            light_instances.append(light_instance)
        return lights, light_instances

    def generate(
        self,
        object_counts: Dict[str, int] = {},
        receptacle_object_counts: Dict[str, Dict[str, int]] = {},
        room_num=None,
    ):
        odb = self.odb
        prefabs = odb.PREFABS
        room_spec = self.room_spec

        log("starting...")
        log(room_spec)

        house_structure = self.generate_structure(room_spec=room_spec)
        interior_boundary = house_structure.interior_boundary
        x_size = interior_boundary.shape[0]
        z_size = interior_boundary.shape[1]

        min_x, min_z, max_x, max_z = (
            0,
            0,
            x_size * self.unit_size,
            z_size * self.unit_size,
        )
        self.placer = RectPlacer((min_x, min_z, max_x, max_z))

        floors_and_walls, door_instances, window_instances = self.add_floors_and_walls(
            house_structure, room_spec, odb, prefabs
        )

        # add light
        # light_prefab = "LowPolyInterior2_Light_04"
        # light_y_size = prefabs[light_prefab]["size"]["y"]
        # floor_instances.append(
        #     {
        #         "prefab": "LowPolyInterior2_Light_04",
        #         "position": [max_x / 2, 3 - light_y_size / 2, max_z / 2],
        #         "rotation": [0, 0, 0],
        #         "scale": [1, 1, 1],
        #         "type": "kinematic",
        #     }
        # )

        floor_polygons = self.get_floor_polygons(house_structure.xz_poly_map)

        self.get_rooms(
            room_type_map=room_spec.room_type_map, floor_polygons=floor_polygons
        )

        # add light
        lights, light_instances = self.get_light()

        floors = house_structure.floorplan
        floors = np.where(floors == 1, 0, floors)
        player, agent = self.add_human_and_agent(floors)

        max_floor_objects = 10

        spawnable_asset_group_info = self.get_spawnable_asset_group_info()

        specified_object_instances = []
        specified_object_types = set()
        if receptacle_object_counts:
            # first place the specified receptacles
            receptacle_type = receptacle
            receptacle = random.choice(odb.OBJECT_DICT[receptacle.lower()])
            specified_object_types.add(odb.OBJECT_TO_TYPE[receptacle])
            count = d["count"]
            prefab_size = odb.PREFABS[receptacle]["size"]
            for _ in range(MAX_SPECIFIED_NUMBER):
                success_flag = False
                for room in self.rooms.values():
                    for _ in range(MAX_SPECIFIED_RECTANGLE_RETRIES):
                        rectangle = room.sample_next_rectangle()
                        minx, minz, maxx, maxz = rectangle
                        rect_x = maxx - minx
                        rect_z = maxz - minz
                        rotation = self.prefab_fit_rectangle(prefab_size, rectangle)

                        if rotation == -1:
                            continue
                        else:
                            x_size = (
                                prefab_size["x"] if rotation == 0 else prefab_size["z"]
                            )
                            z_size = (
                                prefab_size["z"] if rotation == 0 else prefab_size["x"]
                            )
                            minx += x_size / 2 + WALL_THICKNESS
                            minz += z_size / 2 + WALL_THICKNESS
                            maxx -= x_size / 2 + WALL_THICKNESS
                            maxz -= z_size / 2 + WALL_THICKNESS
                            x = np.random.uniform(minx, maxx)
                            z = np.random.uniform(minz, maxz)
                            bbox = (
                                x - x_size / 2,
                                z - z_size / 2,
                                x + x_size / 2,
                                z + z_size / 2,
                            )
                            if self.placer.place_rectangle(receptacle, bbox):
                                specified_object_instances.append(
                                    {
                                        "prefab": receptacle,
                                        "position": [x, prefab_size["y"] / 2, z],
                                        "rotation": [0, rotation, 0],
                                        "scale": [1, 1, 1],
                                        "parent": -1,
                                        "type": "receptacle",
                                        "room_id": room.room_id,
                                        "is_receptacle": True,
                                        "receptacle_type": receptacle_type,
                                    }
                                )
                                log(
                                    f"Specified {receptacle} into position:{ format(x,'.4f')},{format(z,'.4f')}, bbox:{bbox} rotation:{rotation}"
                                )
                                success_flag = True
                                count -= 1
                                break
                    if success_flag:
                        break
                if count == 0:
                    break

        object_instances = []
        for room in self.rooms.values():
            asset = None
            spawnable_asset_groups = spawnable_asset_group_info[
                spawnable_asset_group_info[f"in{room.room_type}s"] > 0
            ]

            floor_types, spawnable_assets = odb.FLOOR_ASSET_DICT[
                (room.room_type, room.split)
            ]

            priority_asset_types = copy.deepcopy(
                odb.PRIORITY_ASSET_TYPES.get(room.room_type, [])
            )
            for i in range(max_floor_objects):
                cache_rectangles = i != 0 and asset is None

                if cache_rectangles:
                    # NOTE: Don't resample failed rectangles
                    # room.last_rectangles.remove(rectangle)
                    rectangle = room.sample_next_rectangle(cache_rectangles=True)
                else:
                    rectangle = room.sample_next_rectangle()

                if rectangle is None:
                    break

                x_info, z_info, anchor_delta, anchor_type = room.sample_anchor_location(
                    rectangle
                )

                asset = self.sample_and_add_floor_asset(
                    room=room,
                    rectangle=rectangle,
                    anchor_type=anchor_type,
                    anchor_delta=anchor_delta,
                    spawnable_assets=spawnable_assets,
                    spawnable_asset_groups=spawnable_asset_groups,
                    priority_asset_types=priority_asset_types,
                    odb=odb,
                )

                if asset is None:
                    continue

                # log(f'asset: {asset}')
                room.sample_place_asset_in_rectangle(
                    asset=asset,
                    rectangle=rectangle,
                    anchor_type=anchor_type,
                    x_info=x_info,
                    z_info=z_info,
                    anchor_delta=anchor_delta,
                )

                added_asset_types = []
                if "assetType" in asset:
                    added_asset_types.append(asset["assetType"])
                else:
                    added_asset_types.extend([o["assetType"] for o in asset["objects"]])

                    if not asset["allowDuplicates"]:
                        spawnable_asset_groups = spawnable_asset_groups.query(
                            f"assetGroupName!='{asset['assetGroupName']}'"
                        )

                for asset_type in added_asset_types:
                    # Remove spawned object types from `priority_asset_types` when appropriate
                    if asset_type in priority_asset_types:
                        priority_asset_types.remove(asset_type)

                    allow_duplicates_of_asset_type = odb.PLACEMENT_ANNOTATIONS.loc[
                        asset_type.lower()
                    ]["multiplePerRoom"]

                    if not allow_duplicates_of_asset_type:
                        # NOTE: Remove all asset groups that have the type
                        spawnable_asset_groups = spawnable_asset_groups[
                            ~spawnable_asset_groups[f"has{asset_type.lower()}"]
                        ]

                        # NOTE: Remove all standalone assets that have the type
                        spawnable_assets = spawnable_assets[
                            spawnable_assets["assetType"] != asset_type
                        ]

        def convert_position(position: Vector3):
            x = a.position["x"]
            y = a.position["y"]
            z = a.position["z"]
            return (x, y, z)

        for room in self.rooms.values():

            log(f"room: {room.room_id}")
            for a in room.assets:
                if isinstance(a, Asset):
                    prefab = a.asset_id
                    prefab_size = odb.PREFABS[prefab]["size"]
                    bbox = (
                        (
                            a.position["x"] - prefab_size["x"] / 2,
                            a.position["z"] - prefab_size["z"] / 2,
                            a.position["x"] + prefab_size["x"] / 2,
                            a.position["z"] + prefab_size["z"] / 2,
                        )
                        if a.rotation == 0 or a.rotation == 180
                        else (
                            a.position["x"] - prefab_size["z"] / 2,
                            a.position["z"] - prefab_size["x"] / 2,
                            a.position["x"] + prefab_size["z"] / 2,
                            a.position["z"] + prefab_size["x"] / 2,
                        )
                    )

                    if not self.placer.place_rectangle(prefab, bbox):
                        log(f"Failed to place{prefab} into {bbox}")
                    elif odb.OBJECT_TO_TYPE[prefab] in specified_object_types:
                        log(f"conflicted with specified objects!")
                    else:
                        log(
                            f"Placed {prefab} into position:{ format(a.position['x'],'.4f')},{format(a.position['z'],'.4f')}, bbox:{bbox} rotation:{a.rotation}"
                        )

                        is_receptacle = True
                        object_instances.append(
                            {
                                "prefab": a.asset_id,
                                "position": convert_position(a.position),
                                "rotation": [0, a.rotation, 0],
                                "scale": [1, 1, 1],
                                "parent": room.room_id,
                                "type": (
                                    "interactable"
                                    if a.asset_id
                                    in self.odb.KINETIC_AND_INTERACTABLE_INFO[
                                        "interactable_names"
                                    ]
                                    else "kinematic"
                                ),
                                "room_id": room.room_id,
                                "is_receptacle": is_receptacle,
                            }
                        )

                else:  # is asset_group
                    assets_dict = a.assets_dict
                    max_bbox = (10000, 100000, -1, -1)
                    asset_group_full_name = []

                    conflict = False
                    for asset in assets_dict:
                        prefab = asset["assetId"]
                        asset_type = odb.OBJECT_TO_TYPE[prefab]
                        if asset_type in specified_object_types:
                            conflict = True
                            break

                        asset_group_full_name.append(prefab)
                        if "children" in asset:
                            for child in asset["children"]:
                                prefab = child["assetId"]
                                asset_group_full_name.append(prefab)
                        prefab_size = odb.PREFABS[prefab]["size"]
                        bbox = (
                            (
                                asset["position"]["x"] - prefab_size["x"] / 2,
                                asset["position"]["z"] - prefab_size["z"] / 2,
                                asset["position"]["x"] + prefab_size["x"] / 2,
                                asset["position"]["z"] + prefab_size["z"] / 2,
                            )
                            if asset["rotation"] == 0 or asset["rotation"] == 180
                            else (
                                asset["position"]["x"] - prefab_size["z"] / 2,
                                asset["position"]["z"] - prefab_size["x"] / 2,
                                asset["position"]["x"] + prefab_size["z"] / 2,
                                asset["position"]["z"] + prefab_size["x"] / 2,
                            )
                        )
                        max_bbox = (
                            min(max_bbox[0], bbox[0]),
                            min(max_bbox[1], bbox[1]),
                            max(max_bbox[2], bbox[2]),
                            max(max_bbox[3], bbox[3]),
                        )
                    asset_group_full_name = "+".join(asset_group_full_name)

                    if not self.placer.place_rectangle(asset_group_full_name, max_bbox):
                        log(f"Failed to place{asset_group_full_name} into {max_bbox}")
                    elif conflict:
                        log(f"conflicted with specified objects!")
                    else:

                        log(f"Placed {asset_group_full_name} into {max_bbox}")
                        for asset in assets_dict:

                            is_receptacle = True
                            if (
                                "tv" in asset["assetId"].lower()
                                or "chair" in asset["assetId"].lower()
                            ):
                                is_receptacle = False

                            object_instances.append(
                                {
                                    "prefab": asset["assetId"],
                                    "position": (
                                        asset["position"]["x"],
                                        asset["position"]["y"],
                                        asset["position"]["z"],
                                    ),
                                    "rotation": [0, asset["rotation"]["y"], 0],
                                    "scale": [1, 1, 1],
                                    "parent": 0,  # 0 represents the floor
                                    "type": (
                                        "interactable"
                                        if asset["assetId"]
                                        in self.odb.KINETIC_AND_INTERACTABLE_INFO[
                                            "interactable_names"
                                        ]
                                        else "kinematic"
                                    ),
                                    "room_id": room.room_id,
                                    "is_receptacle": is_receptacle,
                                }
                            )
                            if "children" in asset:
                                for child in asset["children"]:

                                    is_receptacle = True
                                    if (
                                        "tv" in asset["assetId"].lower()
                                        or "chair" in asset["assetId"].lower()
                                    ):
                                        is_receptacle = False

                                    object_instances.append(
                                        {
                                            "prefab": child["assetId"],
                                            "position": (
                                                child["position"]["x"],
                                                child["position"]["y"],
                                                child["position"]["z"],
                                            ),
                                            "rotation": [0, child["rotation"]["y"], 0],
                                            "scale": [1, 1, 1],
                                            "parent": 0,  # 0 represents the floor
                                            "type": (
                                                "interactable"
                                                if child["assetId"]
                                                in self.odb.KINETIC_AND_INTERACTABLE_INFO[
                                                    "interactable_names"
                                                ]
                                                else "kinematic"
                                            ),
                                            "room_id": room.room_id,
                                            "is_receptacle": is_receptacle,
                                        }
                                    )

        max_object_types_per_room = 10
        small_object_instances = []
        small_object_instances = add_small_objects(
            object_instances,
            odb,
            self.rooms,
            max_object_types_per_room,
            (min_x, min_z, max_x, max_z),
            object_counts=object_counts,
            specified_object_instances=specified_object_instances,
            receptacle_object_counts=receptacle_object_counts,
        )

        ### STEP 5: Adjust Positions for Unity GameObject
        # Convert all the positions (the center of the mesh bounding box) to positions of Unity GameObject transform
        # They are not equal because position of a GameObject also depends on the relative center offset of the mesh within the prefab

        instances = (
            object_instances
            + door_instances
            + window_instances
            + specified_object_instances
            + small_object_instances
        )

        DEBUG = False
        if DEBUG:
            for inst in instances:
                inst["type"] = "kinematic"

        height = max(12, (max_z - min_z) * 1 + 2)
        log(f"min_x: {min_x}, max_x: {max_x}, min_z: {min_z}, max_z: {max_z}")
        center = [(min_x + max_x) / 2, height, (min_z + max_z) / 2]

        room_polygon = []
        for room in self.rooms.values():
            id = room.room_id
            polygon = list(room.room_polygon.polygon.exterior.coords)
            x_center = sum([x for x, _ in polygon]) / len(polygon)
            z_center = sum([z for _, z in polygon]) / len(polygon)
            x_size = max([x for x, _ in polygon]) - min([x for x, _ in polygon])
            z_size = max([z for _, z in polygon]) - min([z for _, z in polygon])
            room_polygon.append(
                {
                    "room_id": id,
                    "room_type": room.room_type,
                    "position": [x_center, 1.5, z_center],
                    "size": [x_size, 3, z_size],
                    "polygon": polygon,
                }
            )
            # print(f'room {id} polygon: {polygon}')

        infos = {
            "prompt": "",
            "instances": instances,
            "player": player,
            "agent": agent,
            "center": center,
            "room_polygon": room_polygon,
        }
        infos.update(floors_and_walls)
        infos["walls"].extend(light_instances)
        infos["lights"] = lights
        with open("last_scene.json", "w", encoding="utf-8") as f:
            json.dump(infos, f, ensure_ascii=False, indent=4)
        return infos
