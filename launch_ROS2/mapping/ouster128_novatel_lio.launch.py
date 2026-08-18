from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os
import yaml


def load_ros1_yaml_as_params(yaml_file_path):
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)

    def flatten_dict(d, parent_key="", sep="/"):
        items = []

        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(
                    flatten_dict(value, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, value))

        return dict(items)

    return flatten_dict(config)


def generate_launch_description():

    package_share = get_package_share_directory("fast_lio_sam")

    config_file = os.path.join(
        package_share,
        "config",
        "mapping",
        "ouster128_novatel.yaml",
    )

    yaml_params = load_ros1_yaml_as_params(config_file)

    fast_lio_params = [
        {
            "use_sim_time": True,

            # First test: validate FAST-LIO frontend alone.
            "sam_enable": True,

            "feature_extract_enable": False,
            "point_filter_num": 3,
            "max_iteration": 3,

            "filter_size_surf": 0.5,
            "filter_size_map": 0.5,
            "cube_side_length": 1000.0,
        },
        yaml_params,
    ]

    fast_lio = Node(
        package="fast_lio_sam",
        executable="fastlio_mapping",
        name="fastlio_mapping",
        output="screen",
        parameters=fast_lio_params,
    )

    return LaunchDescription([
        fast_lio,
    ])