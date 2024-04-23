import os
import requests
import json

class Downloader:
    def __init__(self, download_dir):
      """
        Downloads a file from a given URL and saves it to the specified directory.
        Args:
            url (str): The URL of the file to download.
            file_name (str): The name to save the file as.
        Returns:
            bool: True if the file was successfully downloaded and saved, False otherwise.
        """
        self.download_dir = download_dir

    def download_and_save_json(self, url, filename):
        response = self._make_request(url)
        if response.status_code == 200:
            json_data = response.json()
            file_path = os.path.join(self.download_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(json_data, f)
            print(f"JSON file '{filename}' saved successfully.")
        else:
            print(f"Failed to download JSON from {url}. Status code: {response.status_code}")

    def download_and_save_image(self, url, filename):
        response = self._make_request(url)
        if response.status_code == 200:
            file_path = os.path.join(self.download_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Image '{filename}' saved successfully.")
        else:
            print(f"Failed to download image from {url}. Status code: {response.status_code}")

    def _make_request(self, url):
        try:
            response = requests.get(url)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return None

# Example usage:
download_directory = '/path/to/downloads'
downloader = Downloader(download_directory)

# Download and save JSON file
json_url = 'https://example.com/data.json'
json_filename = 'data.json'
downloader.download_and_save_json(json_url, json_filename)

# Download and save image
image_url = 'https://example.com/image.jpg'
image_filename = 'image.jpg'
downloader.download_and_save_image(image_url, image_filename)
