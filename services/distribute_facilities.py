# TODO:
# happiness_factor -> Dynamic
# Max_dis
# Call Magic Soumik code for min_dist
import json
import time
import random
from copy import deepcopy
import numpy as np
import osmnx as ox
import networkx as nx
from pyproj import Transformer
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

MAX_HAPPINESS = 2
EPSILON = 0.00010
MIN_SAMPLES = 4
np.random.seed(0)

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


def dist_euclidean(point1: Point, point2: Point) -> float:
    return ox.distance.euclidean(point1.y, point1.x, point2.y, point2.x)


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


def calculate_cluster_centroid(cluster, houses) -> dict:
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


def convert_central_points(data: dict) -> dict:
    for key in data["old"]["houses"].keys():
        d["old"]["houses"][key]["central_point"] = Point(
            d["old"]["houses"][key]["central_point"]["long"],
            d["old"]["houses"][key]["central_point"]["lat"],
        )

    for key in data["old"]["facilities"].keys():
        for uuid in data["old"]["facilities"][key].keys():
            x = d["old"]["facilities"][key][uuid]["central_point"]["long"]
            y = d["old"]["facilities"][key][uuid]["central_point"]["lat"]
            d["old"]["facilities"][key][uuid]["central_point"] = Point(x, y)
            (
                d["old"]["facilities"][key][uuid]["node"],
                d["old"]["facilities"][key][uuid]["dist"],
            ) = ox.nearest_nodes(Gc, x, y, return_dist=True)

    if data["new"] != {}:
        data["new"]["central_point"] = Point(
            data["new"]["central_point"]["long"], data["new"]["central_point"]["lat"]
        )

    return data


# Code for optimizing facilities coordinates


# Function to calculate total happiness for a given set of facility coordinates
def calculate_total_happiness(houses, house_nodes, facilities, facility_points):
    total_happiness = 0

    for house_uuid in houses.keys():
        for facility in facilities.keys():
            distance = float("inf")
            for facility_uuid in facilities[facility].keys():
                facility_node = facilities[facility][facility_uuid]["node"]
                if facility_points[facility][1]:
                    new_distance = nx.shortest_path_length(
                        G=Gc,
                        source=house_nodes[house_uuid],
                        target=facility_node,
                        weight="length",
                    )
                    if new_distance < distance:
                        distance = new_distance
                else:
                    point1 = houses[house_uuid]["central_point"]
                    point2 = facilities[facility][facility_uuid]["central_point"]
                    new_distance = dist_euclidean(point1, point2)

                    if new_distance < distance:
                        distance = new_distance

                if distance != float("inf"):
                    if facility_points[facility][1]:
                        if distance > 0:
                            total_happiness += facility_points[facility][0] / distance
                        else:
                            total_happiness += MAX_HAPPINESS
                    else:
                        total_happiness += (
                            facility_points[facility][0] * distance / max_dist
                        )
    avg_happiness = total_happiness / (len(facilities.keys()) * len(houses.keys()))

    print("happiness at this stage: ", avg_happiness)
    time.sleep(1)

    return total_happiness


# Genetic Algorithm to optimize facility coordinates
def optimize_facility_coordinates(houses, facilities, facility_points):
    # Parameters
    population_size = 10
    generations = 10
    mutation_rate = 0.1

    # Precompute House Nodes
    house_nodes = {}
    for house_uuid in houses.keys():
        house_nodes[house_uuid] = ox.nearest_nodes(
            Gc,
            houses[house_uuid]["central_point"].x,
            houses[house_uuid]["central_point"].y,
        )

    # Initial population
    population = []
    for _ in range(population_size):
        individual = deepcopy(facilities)
        for facility in individual.keys():
            for facility_uuid in individual[facility].keys():
                lat = random.uniform(28.4000000, 28.5200000)
                lon = random.uniform(77.6500000, 77.7000000)
                if facility_points[facility][1]:
                    lat = 28.5200000
                    lon = 77.7000000  # max threshold
                individual[facility][facility_uuid]["central_point"] = Point(lon, lat)

        population.append(individual)

    # Main optimization loop
    for generation in range(generations):
        # Evaluate fitness of each individual in the population
        fitness_scores = [
            calculate_total_happiness(houses, house_nodes, ind, facility_points)
            for ind in population
        ]

        # Select the top 50% of individuals based on fitness
        sorted_indices = sorted(
            range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True
        )
        selected_indices = sorted_indices[: population_size // 2]
        selected_population = [population[i] for i in selected_indices]

        # Crossover: Create new individuals by combining features of selected individuals
        new_population = []
        for i in range(population_size // 2):
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            crossover_point = random.randint(0, len(parent1) - 1)

            child = {}
            for facility, coordinates in parent1.items():
                if random.random() < 0.5:
                    child[facility] = deepcopy(parent1[facility])
                else:
                    child[facility] = deepcopy(parent2[facility])

            new_population.append(child)

        # Mutation: Apply random changes to some individuals
        for i in range(population_size // 2):
            if random.random() < mutation_rate:
                mutated_individual = new_population[i]
                for facility in mutated_individual.keys():
                    for facility_uuid in mutated_individual[facility].keys():
                        lat = random.uniform(28.4000000, 28.5200000)
                        lon = random.uniform(77.6500000, 77.7000000)
                        mutated_individual[facility][facility_uuid][
                            "central_point"
                        ] = Point(lon, lat)

        # Combine selected and newly generated individuals
        population = selected_population + new_population

    # Select the best individual from the final population
    final_scores = [
        calculate_total_happiness(houses, house_nodes, ind, facility_points)
        for ind in population
    ]
    best_index = final_scores.index(max(final_scores))
    best_individual = population[best_index]

    return best_individual


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
    transformer = Transformer.from_crs(Gc.graph["crs"], "EPSG:4326")

    max_dist = ox.stats.edge_length_total(Gc)

    with open("../data/facilities.json", "r") as f, open(
        "../data/house.json", "r"
    ) as h:
        facilities_coord = json.load(f)
        houses_coord = json.load(h)

        d = {"old": {"houses": houses_coord, "facilities": facilities_coord}, "new": {}}
        d = convert_central_points(d)

        d["new"] = {
            "key": "uuid",
            "facility": "school",
            "central_point": Point(77.68305, 28.5398),
        }

        start = time.time()
        optimized_facilities = optimize_facility_coordinates(
            d["old"]["houses"], d["old"]["facilities"], facility_points
        )
        end = time.time()
        print("Time taken: ", end - start)

        print("Optimized Facilities:")
        print(optimized_facilities)
        for facility in optimized_facilities.keys():
            for facility_uuid in optimized_facilities[facility].keys():
                print(
                    facility,
                    facility_uuid,
                    optimized_facilities[facility][facility_uuid]["central_point"],
                )
