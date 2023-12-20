import uuid
import osmnx as ox
import json
import subprocess
from os.path import exists

ox.settings.use_cache = True


def fetch_roads_geojson(north: float, south: float, east: float, west: float) -> dict:
    file_name = f"{north}-{south}-{east}-{west}.gpkg"

    if not exists(file_name):
        G = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type="all",
            simplify=True,
            truncate_by_edge=True,
        )
        Gp = ox.project_graph(G)
        ox.io.save_graph_geopackage(Gp, filepath=file_name)

    output = subprocess.check_output(
        [
            f'ogr2ogr -f "GeoJSON" /vsistdout/ -t_srs EPSG:4326 {file_name} edges'
        ],
        shell=True,
    )
    d: dict = json.loads(output)
    return d


def clean_roads_data(data: dict) -> dict:
    roads = {}
    for f in data["features"]:
        if f["geometry"]["type"] == "LineString":
            try:
                if isinstance(f["properties"]["osmid"], str):
                    roads[int(f["properties"]["osmid"])] = f["geometry"]["coordinates"]
                elif isinstance(f["properties"]["osmid"], int):
                    roads[f["properties"]["osmid"]] = f["geometry"]["coordinates"]
                else:
                    roads[f["properties"]["osmid"][0]] = f["geometry"]["coordinates"]
            except Exception as exc:
                print(exc)

    return roads


def fetch_geojson_from_shapefile(
    prj_file: bytes, shape_file: bytes, dbf_file: bytes
) -> dict:
    base_path = uuid.uuid4()
    prj_filepath = f"/tmp/{base_path}.prj"
    shape_filepath = f"/tmp/{base_path}.shp"
    dbf_filepath = f"/tmp/{base_path}.dbf"
    with open(prj_filepath, "wb") as p, open(shape_filepath, "wb") as s, open(
        dbf_filepath, "wb"
    ) as d:
        p.write(prj_file)
        s.write(shape_file)
        d.write(dbf_file)

    output = subprocess.check_output(
        [
            f'SHAPE_RESTORE_SHX=True ogr2ogr -f "GeoJSON" /vsistdout/ -s_srs {prj_filepath} -t_srs EPSG:4326 {shape_filepath}'
        ],
        shell=True,
    )

    return json.loads(output)


def clean_house_data(data: dict) -> dict:
    houses = {}
    for f in data["features"]:
        avg_lon = 0
        avg_lat = 0
        count = len(f["geometry"]["coordinates"][0])

        for point in f["geometry"]["coordinates"][0]:
            avg_lon += point[0]
            avg_lat += point[1]

        avg_lat = avg_lat / count
        avg_lon = avg_lon / count

        houses[str(uuid.uuid4())] = {
            "floors": f["properties"]["No_Floors"],
            "central_point": {"long": avg_lon, "lat": avg_lat},
        }

    return houses


if __name__ == "__main__":
    # fetch_roads(28.5576, 28.5264, 77.7078, 77.6472)
    with open("../data/Builtup_Kalonda.prj", "rb") as p, open(
        "../data/Builtup_Kalonda.shp", "rb"
    ) as s, open("../data/Builtup_Kalonda.dbf", "rb") as d, open(
        "Kalonda_houses.geojson", "w"
    ) as k:
        json.dump(fp=k, obj=fetch_geojson_from_shapefile(p.read(), s.read(), d.read()))
