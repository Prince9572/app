from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

# Replace with your actual API endpoint and key
API_URL = "https://api.example.com/generate"  # Replace with your API URL
API_KEY = "your_api_key_here"  # Replace with your API key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.json.get('prompt')
    response = requests.post(API_URL, json={"prompt": prompt}, headers={"Authorization": f"Bearer {API_KEY}"})
    
    if response.status_code == 200:
        image_url = response.json().get('image_url')  # Adjust based on your API response
        return jsonify({"image_url": image_url})
    else:
        return jsonify({"error": "Image generation failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)