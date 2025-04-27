import requests

API_URL = "http://localhost:8000/process/"
IMAGE_PATH = "path/to/your/input_image.jpg"  # Change this to your image path
OUTPUT_PATH = "output_wireframe.png"         # Where to save the result

with open(IMAGE_PATH, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(API_URL, files=files)

if response.status_code == 200:
    with open(OUTPUT_PATH, "wb") as out_file:
        out_file.write(response.content)
    print(f"Wireframe saved to {OUTPUT_PATH}")
else:
    print("Error:", response.status_code, response.text)