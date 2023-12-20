import json
import numpy as np
import osmnx as ox
import networkx as nx
from pyproj import Transformer
from sklearn.cluster import AgglomerativeClustering
from typing import Tuple

MAX_HAPPINESS = 2
DISTANCE_THRESHOLD = 70
MAX_DISTANCE_TO_B = 1000

facility_points = {
    "administrative": [11, 1],
    "road": [12, 1],
    "school": [13, 1],
    "healthcare": [15, 1],
    "haat_shop_csc": [13, 1],
    "water_facility": [14, 1],
    "electric_facility": [14, 1],
    "solar_plant": [13, 0],
    "biogas": [12, 0],
    "windmill": [13, 0],
    "sanitation": [17, 0],
}


def dist_euclidean(point1: dict, point2: dict) -> float:
    return ox.distance.euclidean(point1["x"], point1["y"], point2["x"], point2["y"])


def cluster_houses(houses_coord: dict):
    coords = np.array(
        [
            (data["central_point"]["x"], data["central_point"]["y"])
            for data in houses_coord.values()
        ]
    )

    houses_coord_keys = list(houses_coord.keys())

    db = AgglomerativeClustering(
        n_clusters=None,
        metric="manhattan",
        linkage="complete",
        distance_threshold=DISTANCE_THRESHOLD,
    ).fit(coords)

    labels = db.labels_.tolist()
    clusters = {k: [] for k in labels}

    for i, label in enumerate(labels):
        clusters[label].append(houses_coord_keys[i])  # Append house UUID

    result_clusters = [
        {uuid: houses_coord[uuid] for uuid in cluster} for cluster in clusters.values()
    ]

    return result_clusters


def calculate_cluster_centroid(cluster, houses) -> dict:
    total_y = 0
    total_x = 0
    for house_uuid in cluster.keys():
        total_x += houses[house_uuid]["central_point"]["x"]
        total_y += houses[house_uuid]["central_point"]["y"]

    centroid = {
        "uuid": f"{list(cluster.keys())[0]}",  # Using the first UUID for the centroid
        "central_point": {
            "x": total_x / len(cluster),
            "y": total_y / len(cluster),
        },
    }

    return centroid


def calculate_happiness(Gc, initial_data: dict) -> Tuple[dict, float, dict]:
    houses = initial_data["old"]["houses"]
    facilities = initial_data["old"]["facilities"]

    if len(houses) == 0:
        raise Exception("No house provided")

    happiness = {facility: 0.0 for facility in facilities.keys()}
    avg_happiness = 0

    house_clusters = cluster_houses(houses)

    for cluster in house_clusters:
        cluster_centroid = calculate_cluster_centroid(cluster, houses)
        cluster_node, cluster_dist = ox.nearest_nodes(
            Gc,
            cluster_centroid["central_point"]["x"],
            cluster_centroid["central_point"]["y"],
            return_dist=True,
        )

        nearest_dist = {}

        for facility_key, facility in facilities.items():
            distance = float("inf")
            uuid = ""
            for facility_uuid in facility.keys():
                point1 = cluster_centroid["central_point"]
                point2 = facility[facility_uuid]["central_point"]
                facility_node = facility[facility_uuid]["node"]

                if facility_points[facility_key][1]:
                    new_distance = (
                        cluster_dist + facility[facility_uuid]["dist"]
                    ) + nx.shortest_path_length(
                        G=Gc, source=cluster_node, target=facility_node, weight="length"
                    )
                else:
                    new_distance = dist_euclidean(point1, point2)

                if new_distance < distance:
                    distance = new_distance
                    uuid = facility_uuid

            nearest_dist[facility_key] = {"id": uuid, "dist": distance}

            if distance != float("inf"):
                if facility_points[facility_key][1]:
                    if distance > 0:
                        happiness[facility_key] = facility_points[facility_key][0] / distance
                    else:
                        happiness[facility_key] = MAX_HAPPINESS
                else:
                    happiness[facility_key] = (
                        facility_points[facility_key][0] * (distance / MAX_DISTANCE_TO_B)
                    )

        for house_uuid in cluster.keys():
            initial_data["old"]["houses"][house_uuid]["nearest_dist"] = nearest_dist

    for facility_key in happiness.keys():
        avg_happiness += happiness[facility_key]

    avg_happiness = avg_happiness / (len(happiness) * len(houses))

    return happiness, avg_happiness, initial_data


def get_nodes_of_facilities(Gc, data: dict) -> dict:
    transformer = Transformer.from_crs(4326, int(Gc.graph["crs"].split(":")[-1]))

    for uuid in data["old"]["houses"].keys():
        x = data["old"]["houses"][uuid]["central_point"]["long"]
        y = data["old"]["houses"][uuid]["central_point"]["lat"]

        x, y = transformer.transform(y, x)

        data["old"]["houses"][uuid]["central_point"]["x"] = x
        data["old"]["houses"][uuid]["central_point"]["y"] = y

    for key in data["old"]["facilities"].keys():
        for uuid in data["old"]["facilities"][key].keys():
            x = data["old"]["facilities"][key][uuid]["central_point"]["long"]
            y = data["old"]["facilities"][key][uuid]["central_point"]["lat"]

            x, y = transformer.transform(y, x)

            data["old"]["facilities"][key][uuid]["central_point"]["x"] = x
            data["old"]["facilities"][key][uuid]["central_point"]["y"] = y

            (
                data["old"]["facilities"][key][uuid]["node"],
                data["old"]["facilities"][key][uuid]["dist"],
            ) = ox.nearest_nodes(Gc, x, y, return_dist=True)

    return data


if __name__ == "__main__":
    ox.__version__
    ox.settings.use_cache = True
    ox.settings.log_console = True

    # G = ox.graph_from_bbox(
    #     north=28.65, south=28.45, east=77.8, west=77.6, network_type="all"
    # )
    # Gp = ox.project_graph(G)
    # Gc = ox.consolidate_intersections(
    #     Gp, rebuild_graph=True, tolerance=20, dead_ends=False
    # )

    # ox.io.save_graphml(Gc, "cache.gml")

    Gc = ox.io.load_graphml("cache.gml")
    transformer = Transformer.from_crs(4326, int(Gc.graph["crs"].split(":")[-1]))

    with open("../data/facilities.json", "r") as f, open(
        "../data/house.json", "r"
    ) as h:
        facilities_coord = json.load(f)
        houses_coord = json.load(h)

        d = {"old": {"houses": houses_coord, "facilities": facilities_coord}}
        d = get_nodes_of_facilities(Gc, d)

        happiness, avg_happiness, d = calculate_happiness(Gc, d)
        print(happiness)
        print(avg_happiness)

        with open("debug.json", "w") as da:
            json.dump(fp=da, obj=d)
