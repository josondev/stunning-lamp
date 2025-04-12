import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import overpy
import networkx as nx
from streamlit_js_eval import get_geolocation
import folium
from streamlit_folium import folium_static
import requests
import polyline
import time
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Set page config
st.set_page_config(page_title="Ambulance Route Helper", layout="wide")

# Function to calculate Haversine distance between two points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r * 1000  # Return meters

# Function to get route between two points using OpenRouteService
def get_route(start_coords, end_coords, api_key, preference="fastest"):
    """Get driving route using OpenRouteService API"""
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    body = {
        "coordinates": [
            [start_coords[1], start_coords[0]],  # OpenRouteService uses [lon, lat]
            [end_coords[1], end_coords[0]]
        ],
        "preference": preference,  # "fastest" or "shortest"
        "profile": "driving-car",  # For ambulance, ideally would be "emergency"
        "format": "geojson"
    }
    
    try:
        response = requests.post(
            'https://api.openrouteservice.org/v2/directions/driving-car/geojson',
            json=body,
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting route: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        st.error(f"Error in API call: {e}")
        return None

# Function to create map with route from OpenRouteService
def create_route_map(user_coords, hospital_coords, hospital_name, api_key, preference="fastest"):
    """Create a folium map with the route from user to hospital"""
    # Create a map centered between the two points
    center_lat = (user_coords[0] + hospital_coords[0]) / 2
    center_lon = (user_coords[1] + hospital_coords[1]) / 2
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add marker for user location
    folium.Marker(
        user_coords,
        popup="Your Location",
        tooltip="Your Location",
        icon=folium.Icon(color="red", icon="user", prefix="fa")
    ).add_to(route_map)
    
    # Add marker for hospital
    folium.Marker(
        hospital_coords,
        popup=hospital_name,
        tooltip=hospital_name,
        icon=folium.Icon(color="green", icon="hospital", prefix="fa")
    ).add_to(route_map)
    
    # Get the route
    route_data = get_route(user_coords, hospital_coords, api_key, preference)
    
    if route_data:
        # Extract route details
        route = route_data['features'][0]
        route_coords = route['geometry']['coordinates']
        # Convert [lon, lat] to [lat, lon] for folium
        route_coords = [[coord[1], coord[0]] for coord in route_coords]
        
        # Extract route summary
        distance = route['properties']['summary']['distance'] / 1000  # Convert to km
        duration = route['properties']['summary']['duration'] / 60    # Convert to minutes
        
        # Add route line to map
        folium.PolyLine(
            route_coords,
            color="blue",
            weight=5,
            opacity=0.7,
            tooltip=f"Distance: {distance:.2f} km, Time: {duration:.1f} min"
        ).add_to(route_map)
        
        # Fit map to route bounds
        route_map.fit_bounds(route_coords)
        
        return route_map, distance, duration
    
    return route_map, None, None

# Function to create unrestricted road network graph
def create_unrestricted_graph(user_coords, radius=5000):
    """Create a graph that ignores traffic restrictions"""
    # Get the road network data
    api = overpy.Overpass()
    query = f"""
        [out:json];
        (
          way["highway"]
            (around:{radius},{user_coords[0]},{user_coords[1]});
        );
        (._;>;);
        out body;
    """
    
    result = api.query(query)
    
    # Create a network graph
    G = nx.DiGraph()
    
    # Add all nodes
    for node in result.nodes:
        G.add_node(node.id, lat=float(node.lat), lon=float(node.lon))
    
    # Add all edges (bidirectional regardless of one-way status)
    for way in result.ways:
        # Get the nodes that make up this way
        nodes = way.nodes
        highway_type = way.tags.get("highway", "unclassified")
        
        # Default speed based on road type (km/h)
        speed_map = {
            "motorway": 120, "trunk": 100, "primary": 90,
            "secondary": 70, "tertiary": 50, "residential": 30,
            "service": 20, "unclassified": 40
        }
        speed = speed_map.get(highway_type, 30)
        
        # Add edges in both directions regardless of one-way status
        for i in range(len(nodes) - 1):
            node1, node2 = nodes[i], nodes[i+1]
            
            # Calculate distance between nodes
            lat1, lon1 = float(node1.lat), float(node1.lon)
            lat2, lon2 = float(node2.lat), float(node2.lon)
            distance = haversine(lon1, lat1, lon2, lat2)
            
            # Calculate travel time (in seconds)
            travel_time = (distance / 1000) / speed * 3600
            
            # Add edges in both directions
            G.add_edge(node1.id, node2.id, distance=distance, time=travel_time)
            G.add_edge(node2.id, node1.id, distance=distance, time=travel_time)
    
    return G

# Function to find nearest node in graph to given coordinates
def find_nearest_node(G, coords):
    """Find the nearest node in the graph to the given coordinates"""
    min_dist = float('inf')
    nearest_node = None
    
    for node in G.nodes():
        node_lat = G.nodes[node]['lat']
        node_lon = G.nodes[node]['lon']
        dist = haversine(coords[1], coords[0], node_lon, node_lat)
        
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node

# Function to find unrestricted shortest path
def find_unrestricted_shortest_path(G, start_node, end_node, weight='distance'):
    """Find shortest path ignoring traffic rules"""
    try:
        # Use Dijkstra's algorithm to find shortest path
        path = nx.shortest_path(G, start_node, end_node, weight=weight)
        path_length = nx.shortest_path_length(G, start_node, end_node, weight=weight)
        
        # Extract coordinates for the path
        path_coords = [(G.nodes[node]['lat'], G.nodes[node]['lon']) for node in path]
        
        return path_coords, path_length
    except nx.NetworkXNoPath:
        return None, None

# Function to create map with unrestricted path
def create_map_with_unrestricted_path(start_coords, end_coords, path_coords, destination_name):
    """Create a folium map with the unrestricted path displayed"""
    # Center map between start and end
    center_lat = (start_coords[0] + end_coords[0]) / 2
    center_lon = (start_coords[1] + end_coords[1]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add markers
    folium.Marker(
        start_coords,
        popup="Your Location",
        tooltip="Your Location",
        icon=folium.Icon(color="red", icon="user", prefix="fa")
    ).add_to(m)
    
    folium.Marker(
        end_coords,
        popup=destination_name,
        tooltip=destination_name,
        icon=folium.Icon(color="green", icon="hospital", prefix="fa")
    ).add_to(m)
    
    # Add path line
    folium.PolyLine(
        path_coords,
        color="red",
        weight=5,
        opacity=0.7,
        tooltip="Unrestricted Path (Ignores Traffic Rules)"
    ).add_to(m)
    
    # Fit map to bounds
    m.fit_bounds([start_coords, end_coords])
    
    return m

# --- Streamlit UI ---
st.title("🚑 Ambulance Route Helper")
st.write("Automatically fetching your current location...")

# Sidebar for API key and settings
with st.sidebar:
    st.header("Settings")
    openrouteservice_api_key = st.text_input(
        "OpenRouteService API Key", 
        value="", 
        type="password",
        help="Get a free API key from https://openrouteservice.org/dev/#/signup"
    )
    
    if not openrouteservice_api_key:
        st.warning("Please enter an OpenRouteService API key to enable routing")
        st.info("Get a free API key from https://openrouteservice.org/dev/#/signup")
    
    routing_type = st.radio(
        "Routing Type",
        ["Standard (Follow Traffic Rules)", "Emergency (Bypass Traffic Rules)"],
        help="Standard follows all traffic rules. Emergency mode ignores one-way streets and other restrictions."
    )
    
    if routing_type == "Emergency (Bypass Traffic Rules)":
        st.warning("⚠️ WARNING: Emergency routing ignores traffic rules and may suggest illegal or dangerous routes. Use at your own risk and only for emergency purposes.")

# --- Step 1: Get user location from browser using JS ---
loc = get_geolocation()

# Initialize session state for selected hospital
if 'selected_hospital' not in st.session_state:
    st.session_state.selected_hospital = None

if 'graph' not in st.session_state:
    st.session_state.graph = None

if loc:
    latitude = loc['coords']['latitude']
    longitude = loc['coords']['longitude']
    st.success(f"📍 Your Location: ({latitude}, {longitude})")

    # --- Step 2: Reverse Geocoding (Get address) ---
    geolocator = Nominatim(user_agent="ambulance_locator")
    location = geolocator.reverse((latitude, longitude), language="en")
    if location:
        st.write(f"📌 Detected Address: {location.address}")
    else:
        st.warning("⚠️ Could not retrieve address from coordinates.")

    # --- Step 3: Fetch Top 5 Nearest Hospitals ---
    user_coords = (latitude, longitude)

    try:
        with st.spinner("Searching for nearby hospitals..."):
            api = overpy.Overpass()
            radius = 5000  # 5 km search radius

            query = f"""
                [out:json];
                node
                  ["amenity"="hospital"]
                  (around:{radius},{latitude},{longitude});
                out;
            """

            result = api.query(query)

            hospitals = []
            for node in result.nodes:
                name = node.tags.get("name", "Unnamed Hospital")
                coords = (float(node.lat), float(node.lon))
                distance = geodesic(user_coords, coords).km
                hospitals.append({
                    "name": name,
                    "coords": coords,
                    "distance": distance
                })

            # Sort and show top 5
            top_hospitals = sorted(hospitals, key=lambda x: x["distance"])[:5]

        if top_hospitals:
            st.write("---")
            st.subheader("🏥 Nearest Hospitals")
            
            # Create two columns - one for hospital list, one for map
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("Click on a hospital to see the route:")
                
                # Display hospitals as buttons
                for i, hospital in enumerate(top_hospitals, start=1):
                    if st.button(f"{i}. {hospital['name']} ({hospital['distance']:.2f} km)"):
                        st.session_state.selected_hospital = hospital
                        
            with col2:
                # If a hospital is selected and we're using standard routing
                if st.session_state.selected_hospital and openrouteservice_api_key and routing_type == "Standard (Follow Traffic Rules)":
                    hospital = st.session_state.selected_hospital
                    
                    with st.spinner(f"Calculating standard route to {hospital['name']}..."):
                        route_map, distance, duration = create_route_map(
                            user_coords, 
                            hospital['coords'], 
                            hospital['name'],
                            openrouteservice_api_key
                        )
                        
                        if distance and duration:
                            st.success(f"Route found! Distance: {distance:.2f} km, Estimated time: {duration:.1f} minutes")
                        
                        folium_static(route_map, width=700)
                
                # If a hospital is selected and we're using emergency routing
                elif st.session_state.selected_hospital and routing_type == "Emergency (Bypass Traffic Rules)":
                    hospital = st.session_state.selected_hospital
                    
                    with st.spinner(f"Calculating emergency route to {hospital['name']}..."):
                        # Create the graph if not already created
                        if st.session_state.graph is None:
                            st.session_state.graph = create_unrestricted_graph(user_coords)
                        
                        # Find nearest nodes to user and hospital
                        user_node = find_nearest_node(st.session_state.graph, user_coords)
                        hospital_node = find_nearest_node(st.session_state.graph, hospital['coords'])
                        
                        # Calculate unrestricted path
                        path_coords, path_length = find_unrestricted_shortest_path(
                            st.session_state.graph, 
                            user_node, 
                            hospital_node
                        )
                        
                        if path_coords and path_length:
                            # Create map with path
                            m = create_map_with_unrestricted_path(
                                user_coords,
                                hospital['coords'],
                                path_coords,
                                hospital['name']
                            )
                            
                            st.success(f"Emergency route found! Distance: {path_length/1000:.2f} km")
                            folium_static(m, width=700)
                        else:
                            st.error("Could not find a path between these locations.")
                
                # If no hospital selected or no API key for standard routing
                elif not st.session_state.selected_hospital or (not openrouteservice_api_key and routing_type == "Standard (Follow Traffic Rules)"):
                    # Create basic map with user location and hospitals
                    m = folium.Map(location=[latitude, longitude], zoom_start=13)
                    
                    # Add marker for user location
                    folium.Marker(
                        [latitude, longitude],
                        popup="Your Location",
                        tooltip="Your Location",
                        icon=folium.Icon(color="red", icon="user", prefix="fa")
                    ).add_to(m)
                    
                    # Add markers for hospitals
                    for hospital in top_hospitals:
                        folium.Marker(
                            hospital["coords"],
                            popup=hospital["name"],
                            tooltip=f"{hospital['name']} - {hospital['distance']:.2f} km",
                            icon=folium.Icon(color="green", icon="hospital", prefix="fa")
                        ).add_to(m)
                    
                    folium_static(m, width=700)
                    
                    if not openrouteservice_api_key and routing_type == "Standard (Follow Traffic Rules)":
                        st.info("Enter an OpenRouteService API key in the sidebar to enable standard routing")
        else:
            st.warning("⚠️ No hospitals found within 5 km.")
    except Exception as e:
        st.error(f"❌ Error while fetching hospitals: {e}")
else:
    st.info("Please allow location access when prompted by your browser.")
    st.info("If you don't see a prompt, check if location access is blocked in your browser settings.")
    
    # Provide option for manual location entry
    st.write("---")
    st.subheader("Or enter your location manually:")
    manual_lat = st.number_input("Latitude", value=28.6139, format="%.6f")
    manual_lon = st.number_input("Longitude", value=77.2090, format="%.6f")
    
    if st.button("Use This Location"):
        # Create a mock location object
        loc = {
            'coords': {
                'latitude': manual_lat,
                'longitude': manual_lon
            }
        }
        st.experimental_rerun()