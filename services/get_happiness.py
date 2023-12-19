import json
import numpy as np
import osmnx as ox
import networkx as nx
from pyproj import Transformer
from sklearn.cluster import AgglomerativeClustering
from typing import Tuple

MAX_HAPPINESS = 1
DISTANCE_THRESHOLD = 50
MAX_DISTANCE_TO_B = 500

facility_points = {
    "administrative": [10, 1],
    "road": [10, 1],
    "school": [15, 1],
    "healthcare": [12, 1],
    "haat_shop_csc": [13, 1],
    "water_facility": [13, 1],
    "electric_facility": [15, 1],
    "solar_plant": [13, 0],
    "biogas": [12, 0],
    "windmill": [13, 0],
    "sanitation": [10, 0],
}


def dist_euclidean(point1: dict, point2: dict) -> float:
    return ox.distance.euclidean(point1["x"], point1["y"], point2["x"], point2["y"])


def cluster_houses(houses_coord):
    coords = np.array(
        [
            (data["central_point"]["x"], data["central_point"]["y"])
            for data in houses_coord.values()
        ]
    )

    db = AgglomerativeClustering(
        n_clusters=None,
        metric="manhattan",
        linkage="complete",
        distance_threshold=DISTANCE_THRESHOLD,
    ).fit(coords)

    labels = db.labels_
    clusters = {}

    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(list(houses_coord.keys())[i])  # Append house UUID

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


def calculate_initial_happiness(initial_data: dict) -> Tuple[dict, float, dict]:
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

        for facility in facilities.keys():
            distance = float("inf")
            uuid = ""
            for facility_uuid in facilities[facility].keys():
                point1 = cluster_centroid["central_point"]
                point2 = facilities[facility][facility_uuid]["central_point"]
                facility_node = facilities[facility][facility_uuid]["node"]

                if facility_points[facility][1]:
                    new_distance = (
                        cluster_dist + facilities[facility][facility_uuid]["dist"]
                    ) + nx.shortest_path_length(
                        G=Gc, source=cluster_node, target=facility_node, weight="length"
                    )

                else:
                    new_distance = dist_euclidean(point1, point2)

                if new_distance < distance:
                    distance = new_distance
                    uuid = facility_uuid

            nearest_dist[facility] = {"id": uuid, "dist": distance}

            if distance != float("inf"):
                if facility_points[facility][1]:
                    if distance > 0:
                        happiness[facility] = facility_points[facility][0] / distance
                    else:
                        happiness[facility] = MAX_HAPPINESS
                else:
                    happiness[facility] = (
                        facility_points[facility][0] * distance / MAX_DISTANCE_TO_B
                    )

        for house_uuid in cluster.keys():
            initial_data["old"]["houses"][house_uuid]["nearest_dist"] = nearest_dist

    for facility in happiness.keys():
        avg_happiness += happiness[facility]

    avg_happiness = avg_happiness / (len(happiness) * len(houses))

    return happiness, avg_happiness, initial_data


def get_nodes_of_facilities(data: dict) -> dict:
    for uuid in data["old"]["houses"].keys():
        x = d["old"]["houses"][uuid]["central_point"]["long"]
        y = d["old"]["houses"][uuid]["central_point"]["lat"]

        x, y = transformer.transform(y, x)

        d["old"]["houses"][uuid]["central_point"]["x"] = x
        d["old"]["houses"][uuid]["central_point"]["y"] = y

    for key in data["old"]["facilities"].keys():
        for uuid in data["old"]["facilities"][key].keys():
            x = d["old"]["facilities"][key][uuid]["central_point"]["long"]
            y = d["old"]["facilities"][key][uuid]["central_point"]["lat"]

            x, y = transformer.transform(y, x)

            d["old"]["facilities"][key][uuid]["central_point"]["x"] = x
            d["old"]["facilities"][key][uuid]["central_point"]["y"] = y

            (
                d["old"]["facilities"][key][uuid]["node"],
                d["old"]["facilities"][key][uuid]["dist"],
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
        d = get_nodes_of_facilities(d)

        happiness, avg_happiness, d = calculate_initial_happiness(d)
        print(happiness)
        print(avg_happiness)

        with open("debug.json", "w") as da:
            json.dump(fp=da, obj=d)
