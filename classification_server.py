#!/usr/bin/env python3
"""
Simple Image Classification Server for Soil/Rock Analysis
Runs on port 5001 with Wikipedia integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import random
import requests
import json

app = Flask(__name__)
CORS(app)

def get_wikipedia_info(rock_type):
    """Get information about rock/soil type from Wikipedia API"""
    try:
        # Search for the rock type on Wikipedia
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + rock_type.replace(" ", "_")
        response = requests.get(search_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "description": data.get("extract", "No description available"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "properties": get_rock_properties(rock_type),
                "uses": get_rock_uses(rock_type)
            }
    except Exception as e:
        print(f"Wikipedia API error: {e}")
    
    # Fallback to local data if API fails
    return get_fallback_rock_info(rock_type)

def get_rock_properties(rock_type):
    """Get properties based on rock type"""
    properties_map = {
        "granite": ["Igneous rock", "High strength", "Crystalline structure", "Weather resistant"],
        "limestone": ["Sedimentary rock", "Medium strength", "Calcium carbonate composition", "Porous"],
        "sandstone": ["Sedimentary rock", "Variable strength", "Quartz-rich", "Stratified"],
        "shale": ["Sedimentary rock", "Low strength", "Fine-grained", "Easily weathered"],
        "basalt": ["Volcanic rock", "High strength", "Dense structure", "Dark colored"],
        "slate": ["Metamorphic rock", "Medium strength", "Foliated structure", "Low porosity"],
        "marble": ["Metamorphic rock", "Medium strength", "Crystalline", "Calcite composition"],
        "quartzite": ["Metamorphic rock", "Very high strength", "Quartz-rich", "Non-foliated"]
    }
    return properties_map.get(rock_type.lower(), ["Variable properties", "Geological analysis needed"])

def get_rock_uses(rock_type):
    """Get common uses based on rock type"""
    uses_map = {
        "granite": ["Construction", "Monuments", "Countertops", "Road aggregate"],
        "limestone": ["Construction", "Cement production", "Agriculture", "Chemical industry"],
        "sandstone": ["Construction", "Paving", "Decorative stone", "Filtration"],
        "shale": ["Brick making", "Cement production", "Oil and gas extraction"],
        "basalt": ["Construction", "Road base", "Concrete aggregate", "Railroad ballast"],
        "slate": ["Roofing", "Flooring", "Billiard tables", "Electrical panels"],
        "marble": ["Sculpture", "Architecture", "Decorative elements", "Lime production"],
        "quartzite": ["Construction", "Railroad ballast", "Roofing granules", "Glass making"]
    }
    return uses_map.get(rock_type.lower(), ["General construction", "Industrial applications"])

def get_fallback_rock_info(rock_type):
    """Fallback geological information when Wikipedia API is unavailable"""
    fallback_data = {
        "granite": {
            "description": "Granite is a coarse-grained igneous rock composed of quartz, feldspar, and mica. It forms from the slow crystallization of magma below Earth's surface.",
            "properties": ["Igneous rock", "High compressive strength", "Crystalline structure", "Weather resistant"],
            "uses": ["Construction", "Monuments", "Countertops", "Road aggregate"],
            "url": "https://en.wikipedia.org/wiki/Granite"
        },
        "limestone": {
            "description": "Limestone is a sedimentary rock composed mainly of calcium carbonate. It often forms from the remains of marine organisms.",
            "properties": ["Sedimentary rock", "Medium strength", "Calcium carbonate composition", "Porous structure"],
            "uses": ["Construction", "Cement production", "Agriculture", "Chemical industry"],
            "url": "https://en.wikipedia.org/wiki/Limestone"
        },
        "sandstone": {
            "description": "Sandstone is a sedimentary rock composed of sand-sized minerals or rock grains, primarily quartz.",
            "properties": ["Sedimentary rock", "Variable strength", "Quartz-rich", "Stratified appearance"],
            "uses": ["Construction", "Paving stones", "Decorative stone", "Filtration media"],
            "url": "https://en.wikipedia.org/wiki/Sandstone"
        }
    }
    
    return fallback_data.get(rock_type.lower(), {
        "description": f"{rock_type} is a geological material with varying properties depending on its formation and composition.",
        "properties": ["Variable properties", "Requires detailed geological analysis"],
        "uses": ["General construction", "Industrial applications"],
        "url": ""
    })

@app.route('/classify', methods=['POST'])
def classify_rock_image():
    """Enhanced classification with Wikipedia integration"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        # Simulate classification (in real app, this would use ML model)
        rock_types = ['granite', 'limestone', 'sandstone', 'shale', 'basalt', 'slate', 'marble', 'quartzite']
        predicted_class = random.choice(rock_types)
        confidence = random.randint(75, 95)
        
        # Get Wikipedia information
        wikipedia_info = get_wikipedia_info(predicted_class)
        
        # Generate geological analysis
        geological_details = {
            "formation_type": random.choice(["Igneous", "Sedimentary", "Metamorphic"]),
            "hardness": f"{random.randint(3, 8)}/10 Mohs scale",
            "porosity": random.choice(["Low", "Medium", "High"]),
            "mining_suitability": random.choice(["Excellent", "Good", "Fair", "Poor"])
        }
        
        # Risk analysis
        risk_levels = ["Low", "Medium", "High"]
        risk_level = random.choice(risk_levels)
        
        analysis = {
            "risk_level": risk_level,
            "stability_assessment": f"{random.choice(['Stable', 'Moderately stable', 'Unstable'])} under current conditions",
            "recommendations": f"Monitor for {random.choice(['weathering', 'structural changes', 'erosion patterns'])} and implement appropriate safety measures."
        }
        
        return jsonify({
            "predicted_class": predicted_class.title(),
            "confidence": confidence,
            "wikipedia_info": wikipedia_info,
            "geological_details": geological_details,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Image Classification API",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🎯 Starting Image Classification Server...")
    print("🔗 Classification API: http://localhost:5001/classify")
    print("❤️  Health Check: http://localhost:5001/health")
    
    app.run(host='0.0.0.0', port=5001, debug=True)