import requests

# URL of the FastAPI server
url = "http://localhost:8000/predict/"

# Path to the image file you want to predict
image_path = "6.png"

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Send the request
    response = requests.post(url, files={"file": image_file})

# Print the response from the server
print(response.json())
