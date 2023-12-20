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
from sklearn.cluster import AgglomerativeClustering

MAX_HAPPINESS = 2
DISTANCE_THRESHOLD = 70

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


def dist_euclidean(point1: dict, point2: dict) -> float:
    return ox.distance.euclidean(point1["y"], point1["x"], point2["y"], point2["x"])


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
    total_lat = 0
    total_lon = 0
    for house_uuid in cluster.keys():
        total_lat += houses[house_uuid]["central_point"]["y"]
        total_lon += houses[house_uuid]["central_point"]["x"]

    centroid = {
        "uuid": f"{list(cluster.keys())[0]}",  # Using the first UUID for the centroid
        "central_point": {  # Using the average of all the points for the centroid
            "x": total_lon / len(cluster.keys()),
            "y": total_lat / len(cluster.keys()),
        },
    }

    print("centroid: ", centroid)

    return centroid


# Code for optimizing facilities coordinates


# Function to calculate total happiness for a given set of facility coordinates
def calculate_total_happiness(cluster_data, facilities, facility_points):
    total_happiness = 0

    for cluster_uuid in cluster_data.keys():
        for facility in facilities.keys():
            distance = float("inf")
            for facility_uuid in facilities[facility].keys():
                facility_node = facilities[facility][facility_uuid]["node"]
                if facility_points[facility][1]:
                    new_distance = nx.shortest_path_length(
                        G=Gc,
                        source=cluster_data[cluster_uuid]["node"],
                        target=facility_node,
                        weight="length",
                    )
                    if new_distance < distance:
                        distance = new_distance
                else:
                    point1 = cluster_data[cluster_uuid]["central_point"]
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
    avg_happiness = total_happiness / (
        len(facilities.keys()) * len(cluster_data.keys())
    )
    print("len of cluster_data: ", len(cluster_data.keys()))
    print("len of facilities: ", len(facilities.keys()))

    print("happiness at this stage: ", avg_happiness)
    time.sleep(1)

    return avg_happiness


# Genetic Algorithm to optimize facility coordinates
def optimize_facility_coordinates(houses, facilities, facility_points):
    # Parameters
    population_size = 10
    generations = 10
    mutation_rate = 0.1

    # Precomputed Cluster Nodes
    cluster_data = {}
    house_clusters = cluster_houses(houses)
    for cluster in house_clusters:
        cluster_centroid = calculate_cluster_centroid(cluster, houses)
        cluster_data[cluster_centroid["uuid"]] = {
            "node": ox.nearest_nodes(
                Gc,
                cluster_centroid["central_point"]["x"],
                cluster_centroid["central_point"]["y"],
                return_dist=False,
            ),
            "central_point": cluster_centroid["central_point"],
        }

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
                individual[facility][facility_uuid]["central_point"]["x"] = lon
                individual[facility][facility_uuid]["central_point"]["y"] = lat

        population.append(individual)

    # Main optimization loop
    for generation in range(generations):
        # Evaluate fitness of each individual in the population
        fitness_scores = [
            calculate_total_happiness(cluster_data, ind, facility_points)
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
                        mutated_individual[facility][facility_uuid]["central_point"][
                            "x"
                        ] = lon
                        mutated_individual[facility][facility_uuid]["central_point"][
                            "y"
                        ] = lat

        # Combine selected and newly generated individuals
        population = selected_population + new_population

    # Select the best individual from the final population
    final_scores = [
        calculate_total_happiness(cluster_data, ind, facility_points)
        for ind in population
    ]
    best_index = final_scores.index(max(final_scores))
    best_individual = population[best_index]

    return best_individual


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
        d = get_nodes_of_facilities(Gc, d)

        # d["new"] = {
        #     "key": "uuid",
        #     "facility": "school",
        #     "central_point": Point(77.68305, 28.5398),
        # }

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
