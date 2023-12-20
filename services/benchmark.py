import osmnx as ox
import networkx as nx
from geopandas import GeoSeries
from shapely.geometry import Point
import random
import time


def func():
    G = ox.graph_from_bbox(
        north=28.65, south=28.45, east=77.8, west=77.6, network_type="all"
    )
    Gp = ox.project_graph(G)
    Gc = ox.consolidate_intersections(
        Gp, rebuild_graph=True, tolerance=20, dead_ends=True
    )

    start = time.time()

    start_points = [
        Point(77.6 + random.random() * 0.2, 28.45 + random.random() * 0.2)
        for _ in range(1000)
    ]

    end_points = [
        Point(77.6 + random.random() * 0.2, 28.45 + random.random() * 0.2)
        for _ in range(1000)
    ]

    start_series = GeoSeries(data=start_points, crs="wgs84").to_crs(epsg="32643")
    end_series = GeoSeries(data=end_points, crs="wgs84").to_crs(epsg="32643")

    start_nodes = ox.nearest_nodes(Gc, [pt.x for pt in start_series], [pt.y for pt in start_series])
    end_nodes = ox.nearest_nodes(Gc, [pt.x for pt in end_series], [pt.y for pt in end_series])
    print(start_nodes)
    print(end_nodes)

    mid = time.time()
    print(mid - start)
    print(len(start_nodes))
    ox.shortest_path(Gc, start_nodes, end_nodes, weight="length")

    end = time.time()
    print(end - start)
    print(end - mid)


func()
