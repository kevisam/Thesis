import xml.etree.ElementTree as ET
import networkx as nx

# Parse the XML data

# Specify the path to your OSM file
osm_file_path = "TestMap.osm"

# Read the content of the OSM file
with open(osm_file_path, "r", encoding="utf-8") as file:
    osm_data = file.read()

root = ET.fromstring(osm_data)

# Create a directed graph using NetworkX
G = nx.DiGraph()

# Extract nodes and add them to the graph
for node_elem in root.findall(".//node"):
    node_id = int(node_elem.get("id"))
    lat = float(node_elem.get("lat"))
    lon = float(node_elem.get("lon"))
    G.add_node(node_id, pos=(lat, lon))

# Create edges with an initial weight of 1
for way_elem in root.findall(".//way"):
    way_nodes = way_elem.findall(".//nd")
    for i in range(len(way_nodes) - 1):
        node1 = int(way_nodes[i].get("ref"))
        node2 = int(way_nodes[i + 1].get("ref"))
        G.add_edge(node1, node2, weight=1)

# Visualization (optional)
import matplotlib.pyplot as plt

pos = nx.get_node_attributes(G, "pos")
nx.draw(G, pos, with_labels=False, font_size=8, node_size=10)
plt.show()
