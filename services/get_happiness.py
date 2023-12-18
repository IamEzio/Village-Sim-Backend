# TODO:
# happiness_factor -> Dynamic
# Max_dis
# Call Magic Soumik code for min_dist
import json
import time
import numpy as np
import osmnx as ox
import networkx as nx
from geopandas import GeoSeries
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from typing import Tuple

MAX_HAPPINESS = 2
EPSILON = 0.00015
MIN_SAMPLES = 10
np.random.seed(0)

facility_points = {
    "administrative": [10, 1],
    "road": [10, 1],
    "school": [15, 1],
    "healthcare": [12, 1],
    "haat_shop_csc": [13, 1],
    "water_facility": [13, 1],
    "electric_facility": [15, 1],
    "solar_plant": [13, -1],
    "biogas": [12, -1],
    "windmill": [13, -1],
    "sanitation": [10, -1],
}


def dist_road(point1: Point, point2: Point) -> float:
    print("Point1: ", point1)
    print("Point2: ", point2)
    long_lats = [point1, point2]
    points = GeoSeries(data=long_lats, crs="wgs84")
    points_proj = points.to_crs(epsg="32643")
    nodes = []
    for pt in points_proj:
        nodes.append(ox.nearest_nodes(Gc, pt.x, pt.y))
    print(nodes)
    # route = ox.shortest_path(Gc, nodes[0], nodes[1], weight="length")
    route_length = nx.shortest_path_length(
        G=Gc, source=nodes[0], target=nodes[1], weight="length"
    )
    print(route_length)
    return route_length


def dist_euclidean(point1: Point, point2: Point) -> float:
    return ox.distance.euclidean(point1.y, point1.x, point2.y, point2.x)


def calculate_happiness_for_neighborhood(
    clustercentroid, facilities, facility, happiness, max_dist
) -> Tuple[dict, dict]:
    nearest_dist = {}
    distance = float("inf")
    uuid = ""

    for facility_uuid in facilities[facility].keys():
        point1 = clustercentroid["central_point"]
        point2 = facilities[facility][facility_uuid]["central_point"]

        if facility_points[facility][1]:
            new_distance = dist_road(point1, point2)
            if new_distance < distance:
                distance = new_distance
                uuid = facility_uuid
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
            happiness[facility] = facility_points[facility][0] * distance / max_dist

    return happiness, nearest_dist


def cluster_houses(houses_coord, epsilon=EPSILON, min_samples=MIN_SAMPLES):
    coords = [
        (data["central_point"].x, data["central_point"].y)
        for data in houses_coord.values()
    ]

    # Using DBSCAN for clustering
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coords)
    labels = db.labels_

    clusters = {}
    print("labels: ", labels)
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(list(houses_coord.keys())[i])  # Append house UUID

    print("clusters: ", len(clusters))

    result_clusters = [
        {uuid: houses_coord[uuid] for uuid in cluster} for cluster in clusters.values()
    ]
    print("res: ", result_clusters)

    return result_clusters


def calculate_cluster_centroid(cluster, houses):
    total_lat = 0
    total_lon = 0
    for house_uuid in cluster.keys():
        total_lat += houses[house_uuid]["central_point"].y
        total_lon += houses[house_uuid]["central_point"].x

    centroid = {
        "uuid": f"{list(cluster.keys())[0]}",  # Using the first UUID for the centroid
        "central_point": Point(total_lon / len(cluster), total_lat / len(cluster)),
    }

    print("centroid: ", centroid)

    return centroid


def calculate_initial_happiness(initial_data: dict) -> Tuple[dict, float, dict]:
    houses = initial_data["old"]["houses"]

    if len(houses) == 0:
        raise Exception("No house provided")

    facilities = initial_data["old"]["facilities"]

    happiness = {}
    for facility in facilities.keys():
        happiness[facility] = 0
    avg_happiness = 0

    # for house_uuid in houses.keys():
    #     nearest_dist = {}
    #     for facility in facilities.keys():
    #         distance = float("inf")
    #         uuid = ""

    #         for facility_uuid in facilities[facility].keys():
    #             point1 = houses[house_uuid]["central_point"]
    #             point2 = facilities[facility][facility_uuid]["central_point"]

    #             if facility_points[facility][1]:
    #                 new_distance = dist_road(point1, point2)

    #                 if new_distance < distance:
    #                     distance = new_distance
    #                     uuid = facility_uuid
    #             else:
    #                 new_distance = dist_euclidean(point1, point2)

    #                 if new_distance < distance:
    #                     distance = new_distance
    #                     uuid = facility_uuid

    #         nearest_dist[facility] = {"id": uuid, "dist": distance}

    #         if distance != float("inf"):
    #             if facility_points[facility][1]:
    #                 if distance > 0:
    #                     happiness[facility] += facility_points[facility][0] / distance
    #                 else:
    #                     happiness[facility] = MAX_HAPPINESS
    #             else:
    #                 happiness[facility] += (
    #                     facility_points[facility][0] * distance / max_dist
    #                 )

    #     initial_data["old"]["houses"][house_uuid]["nearest_dist"] = nearest_dist

    start = time.time()

    house_clusters = cluster_houses(houses)

    mid = time.time()

    f = 0
    a = []
    for cluster in house_clusters:
        cluster_centroid = calculate_cluster_centroid(cluster, houses)
        nearest_dist = {}

        for facility in facilities.keys():
            distance = float("inf")
            uuid = ""
            for facility_uuid in facilities[facility].keys():
                point1 = cluster_centroid["central_point"]
                point2 = facilities[facility][facility_uuid]["central_point"]

                if facility_points[facility][1]:
                    new_distance = dist_road(point1, point2)
                    if new_distance < distance:
                        distance = new_distance
                        uuid = facility_uuid
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
                        facility_points[facility][0] * distance / max_dist
                    )
        s = 0
        for house_uuid in cluster.keys():
            f += 1
            s += 1
            initial_data["old"]["houses"][house_uuid]["nearest_dist"] = nearest_dist
        a.append(s)

    minpos = a.index(min(a))

    end = time.time()
    print("Time taken: ", end - start)
    print("Time taken for clustering: ", mid - start)
    print("Total houses ", len(houses))
    print("Total house clusters: ", len(house_clusters))
    print("Total houses in all clusters: ", f)
    print("Total houses in each cluster: ", a)
    # print("Cluster with minimum houses: ", minpos, " ", house_clusters[minpos])

    for house_data in house_clusters[minpos].values():
        print("House coordinates: ", house_data["central_point"])
    # get average of all happiness
    for facility in happiness.keys():
        avg_happiness += happiness[facility]

    avg_happiness = avg_happiness / (len(happiness) * len(houses))

    return happiness, avg_happiness, initial_data


def calculate_updated_happiness_on_adding_facility(
    data, happiness
) -> Tuple[dict, float, dict]:
    houses = data["old"]["houses"]
    new_building = data["new"]["facility"]
    new_building_coord = data["new"]["central_point"]

    if len(houses) == 0:
        raise Exception("No House")
    for house_uuid in houses.keys():
        curr_house_coordinates = houses[house_uuid]["central_point"]
        old_distance = houses[house_uuid]["nearest_dist"].get(
            new_building, {"dist": 0}
        )["dist"]

        if facility_points[new_building][1]:
            new_distance = dist_road(new_building_coord, curr_house_coordinates)

            if new_distance < old_distance:
                if old_distance != float("inf"):
                    if old_distance > 0:
                        happiness[new_building] -= (
                            facility_points[new_building][0] / old_distance
                        )
                    else:
                        happiness[new_building] -= MAX_HAPPINESS
                if new_distance > 0:
                    happiness[new_building] += (
                        facility_points[new_building][0] / new_distance
                    )
                else:
                    happiness[new_building] += MAX_HAPPINESS

        else:
            new_distance = dist_euclidean(new_building_coord, curr_house_coordinates)

            if new_distance < old_distance:
                if old_distance != float("inf"):
                    happiness[new_building] -= (
                        facility_points[new_building][0] * old_distance / max_dist
                    )
                happiness[new_building] += (
                    facility_points[new_building][0] * new_distance / max_dist
                )

    avg_happiness = 0
    for facility in happiness.keys():
        avg_happiness += happiness[facility]

    avg_happiness = avg_happiness / (len(happiness) * len(houses))

    return happiness, avg_happiness, data


def calculate_updated_happiness_on_adding_house(
    data: dict, happiness: dict
) -> Tuple[dict, float, dict]:
    facilities_coord = data["old"]["facilities"]
    new_building_coord = data["new"]["central_point"]

    nearest_distances = {}
    for facility in facilities_coord.keys():
        if facility_points[facility][1]:
            distance = float("inf")
            for facility_uuid in facilities_coord[facility].keys():
                curr_facility_coordinates = facilities_coord[facility][facility_uuid][
                    "central_point"
                ]
                distance = min(
                    dist_road(new_building_coord, curr_facility_coordinates),
                    distance,
                )
            if distance > 0:
                happiness[facility] += facility_points[facility][0] / distance
            else:
                happiness[facility] += MAX_HAPPINESS

        else:
            distance = float("inf")
            for facility_uuid in facilities_coord[facility].keys():
                curr_facility_coordinates = facilities_coord[facility][facility_uuid][
                    "central_point"
                ]
                distance = min(
                    dist_euclidean(new_building_coord, curr_facility_coordinates),
                    distance,
                )
            happiness[facility] += facility_points[facility][0] * distance / max_dist

        nearest_distances[facility] = {"id": "-1", "dist": distance}

    avg_happiness = 0
    for facility in happiness.keys():
        avg_happiness += happiness[facility]

    avg_happiness = avg_happiness / (len(happiness) * len(houses_coord))

    return happiness, avg_happiness, nearest_distances


def convert_central_points(data: dict) -> dict:
    for key in data["old"]["houses"].keys():
        d["old"]["houses"][key]["central_point"] = Point(
            d["old"]["houses"][key]["central_point"]["long"],
            d["old"]["houses"][key]["central_point"]["lat"],
        )

    for key in data["old"]["facilities"].keys():
        for uuid in data["old"]["facilities"][key].keys():
            d["old"]["facilities"][key][uuid]["central_point"] = Point(
                d["old"]["facilities"][key][uuid]["central_point"]["long"],
                d["old"]["facilities"][key][uuid]["central_point"]["lat"],
            )

    if data["new"] != {}:
        data["new"]["central_point"] = Point(
            data["new"]["central_point"]["long"], data["new"]["central_point"]["lat"]
        )

    return data


if __name__ == "__main__":
    ox.__version__
    ox.settings.use_cache = True
    ox.settings.log_console = True

    G = ox.graph_from_bbox(
        north=28.65, south=28.45, east=77.8, west=77.6, network_type="all"
    )
    Gp = ox.project_graph(G)
    Gc = ox.consolidate_intersections(
        Gp, rebuild_graph=True, tolerance=20, dead_ends=False
    )

    ox.io.save_graphml(Gc, "cache.gml")

    Gc = ox.io.load_graphml("cache.gml")

    max_dist = ox.stats.edge_length_total(Gc)

    with open("../data/facilities.json", "r") as f, open(
        "../data/house.json", "r"
    ) as h:
        facilities_coord = json.load(f)
        houses_coord = json.load(h)

        d = {"old": {"houses": houses_coord, "facilities": facilities_coord}, "new": {}}
        d = convert_central_points(d)

        happiness, avg_happiness, d = calculate_initial_happiness(d)
        print(happiness)
        print(avg_happiness)
        print(d)

        d["new"] = {
            "key": "uuid",
            "facility": "school",
            "central_point": Point(77.68305, 28.5398),
        }

        happiness, avg_happiness, d = calculate_updated_happiness_on_adding_facility(
            d, happiness
        )
