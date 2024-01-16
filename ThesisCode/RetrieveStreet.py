from geopy.geocoders import Nominatim


# def parse_gpgga(gpgga_data):

#     #TODO

#     return latitude, longitude


def get_street_name(latitude, longitude):
    # Initialize the geolocator
    geolocator = Nominatim(user_agent="street_name_locator")

    # Construct the location tuple
    location = geolocator.reverse((latitude, longitude), language="en")

    # Extract street name from the location address
    street_name = location.raw.get("address", {}).get("road", None)

    return street_name


import xml.etree.ElementTree as ET


def find_edge_by_street(osm_file, target_street_name):
    tree = ET.parse(osm_file)
    root = tree.getroot()

    for way in root.findall(".//way"):
        street_name = None

        # Check if the way has a tag with key "name" and value equal to the target street name
        for tag in way.findall("./tag"):
            if tag.get("k") == "name" and tag.get("v") == target_street_name:
                street_name = target_street_name
                break

        if street_name:
            # Retrieve the nodes of the way
            nodes = [nd.get("ref") for nd in way.findall("./nd")]

            # Assuming an edge is represented by a sequence of nodes
            edge = {"street_name": street_name, "nodes": nodes}

            return edge

    return None


# Example usage

# Example coordinates (50.845797, 4.455772) of Albert Dumontlaan
latitude = 50.845797
longitude = 4.455772

# Get the street name
street_name = get_street_name(latitude, longitude)

if street_name:
    print(f"The street name at coordinates ({latitude}, {longitude}) is: {street_name}")
else:
    print(f"No street name found for coordinates ({latitude}, {longitude})")

osm_file_path = "TestMap.osm"

result = find_edge_by_street(osm_file_path, street_name)

if result:
    print(f"Edge with street name '{street_name}' found. Nodes: {result['nodes']}")
else:
    print(f"No edge found with street name '{street_name}' in the provided OSM file.")
