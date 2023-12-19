import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from services.facilities import fetch_facilities
from services.get_buildings_data import get_buildings_data
from services.get_happiness import (
    calculate_initial_happiness,
    calculate_updated_happiness_on_adding_facility,
)
from services.constants import fetch_constants, update_constants
import requests
import pandas as pd

from services.roads_shapefile import (
    clean_house_data,
    clean_roads_data,
    fetch_geojson_from_shapefile,
    fetch_roads_geojson,
)

app = Flask(__name__)
CORS(app)


@app.route("/residential/kalonda")
def get_kalonda_residential_data():
    residential_data = get_buildings_data("data/Builtup_Kalonda.geojson")
    return json.dumps(residential_data)


@app.route("/facilities", methods=["GET"])
def get_facilities():
    args = request.args
    gpcode = args.get("gpcode", default=63317, type=int)
    response = requests.get(
        f"https://egramswaraj.gov.in/webservice/getGisMapAsset/{gpcode}"
    )
    return fetch_facilities(response.json())


@app.route("/roads", methods=["GET"])
def get_roads():
    args = request.args
    north = args.get("north", default=28.58, type=float)
    south = args.get("south", default=(north - 0.12), type=float)
    east = args.get("east", default=77.72, type=float)
    west = args.get("west", default=(east + 0.12), type=float)

    geojson = fetch_roads_geojson(north, south, east, west)
    return clean_roads_data(geojson)


@app.route("/houses/shapefile", methods=["POST"])
def get_houses_from_shapefile():
    shp = request.files["shp"].read()
    prj = request.files["prj"].read()
    dbf = request.files["dbf"].read()

    geojson = fetch_geojson_from_shapefile(prj, shp, dbf)
    return clean_house_data(geojson)


@app.route("/interchange", methods=["POST"])
def get_facilities_houses():
    # houses
    shp = request.files["shp"].read()
    prj = request.files["prj"].read()
    dbf = request.files["dbf"].read()

    geojson = fetch_geojson_from_shapefile(prj, shp, dbf)
    houses = clean_house_data(geojson)

    # facilities
    args = request.args
    gpcode = args.get("gpcode", default=63317, type=int)
    response = requests.get(
        f"https://egramswaraj.gov.in/webservice/getGisMapAsset/{gpcode}"
    )
    facilities = fetch_facilities(response.json())

    return {"old": {"houses": houses, "facilities": facilities}, "new": {}}


@app.route("/get_initial_happiness", methods=["POST"])
def get_initial_happiness():
    try:
        request_data = request.get_json()
        result = calculate_initial_happiness(request_data)
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})


@app.route("/get_updated_happiness", methods=["POST"])
def get_updated_happiness_on_adding_house():
    try:
        request_data = request.get_json()
        data = request_data["data"]
        happiness = request_data["happiness"]
        result = calculate_updated_happiness_on_adding_facility(data, happiness)
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})


@app.route("/constants", methods=["GET"])
def get_constants():
    constants_data = fetch_constants()
    return jsonify(constants_data)


@app.route("/constants", methods=["POST"])
def update_constants_route():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON data provided"}), 400

    result = update_constants(data)
    return jsonify(result)


@app.route("/get_local_bodies", methods=["GET"])
def get_local_bodies():
    compressed_csv_file_path = "data/selected_localbodies.csv.gz"
    df = pd.read_csv(compressed_csv_file_path)
    try:
        state_code = request.args.get("stateCode")
        local_body_type_code = request.args.get("localBodyTypeCode")

        filtered_df = df
        if state_code:
            filtered_df = filtered_df[filtered_df["stateCode"] == int(state_code)]
        if local_body_type_code:
            filtered_df = filtered_df[filtered_df["localBodyTypeCode"] == int(local_body_type_code)]

        result = filtered_df.to_dict(orient="records")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_panchayats/<statecode>", methods=["GET"])
def get_panchayats(statecode):
    compressed_csv_file_path = "data/selected_localbodies.csv.gz"
    df = pd.read_csv(compressed_csv_file_path)
    try:
        state_code = int(statecode)

        filtered_df = df[df["stateCode"] == state_code]

        panchayats_data = filtered_df[["coverage_entityName", "localBodyCode"]]

        # Convert the result to a dictionary
        panchayats_dict = panchayats_data.to_dict(orient="records")

        return jsonify({"panchayats": panchayats_dict})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
