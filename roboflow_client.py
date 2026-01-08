"""Client for interacting with Roboflow REST API."""

from typing import Dict, List
import requests


class RoboflowAPIClient:
    """Client for interacting with Roboflow REST API."""
    
    def __init__(self, api_key: str, workspace_id: str, project_id: str):
        # Store API credentials and identifiers
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.project_id = project_id
        # Construct base URL for API requests
        self.base_url = f"https://api.roboflow.com/{workspace_id}/{project_id}"
        # Set authorization header for authenticated requests
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_images_list(self) -> List[Dict]:
        """Fetch list of all images in the project using the search endpoint."""
        try:
            # Use the project-level search endpoint with API key in query parameter
            search_url = f"{self.base_url}/search?api_key={self.api_key}"
            all_images = []
            limit = 100  # Number of images per page
            offset = 0
            
            while True:

                # POST request to search endpoint with in_dataset filter
                # Body only contains in_dataset as specified
                payload = {
                    "in_dataset": True,
                    "limit": limit,
                    "offset": offset
                }
                response = requests.post(
                    search_url, 
                    headers={"Content-Type": "application/json"}, 
                    json=payload,
                    timeout=30
                )
                # Raise exception if HTTP status indicates error
                response.raise_for_status()
                # Parse JSON response
                data = response.json()
                # Extract images from results field
                images = data.get("results", [])
                if not images:
                    # No more images to fetch
                    break
                all_images.extend(images)
                
                # Check pagination - get total
                total = data.get("total", 0)
                print(f"Total images: {total}")
                print(f"Fetched {len(all_images)} images so far")
                
                # Check if we've fetched all images
                if total > 0 and len(all_images) >= total:
                        # Fetched all images
                        break
                elif len(images) < limit:
                    # Last page, no more images
                    break
                # Move to next page
                offset += limit
            
            return all_images
        except requests.exceptions.RequestException as e:
            print(f"Error fetching images list: {e}")
            raise
    
    def get_image_metadata(self, image_id: str) -> Dict:
        """Fetch full metadata for a specific image including annotations."""
        # Construct URL for specific image tags endpoint with API key in query parameter
        image_url = f"{self.base_url}/images/{image_id}?api_key={self.api_key}"
        try:
            # Get request to fetch detailed metadata including annotations
            response = requests.get(
                image_url, 
                headers={"Content-Type": "application/json"}, 
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            # Response is in format {"image": {...}}, extract the image object
            return data.get("image", data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image metadata for {image_id}: {e}")
            raise
    
    def download_image(self, image_url: str) -> bytes:
        """Download image from URL."""
        try:
            # Download raw image bytes from URL
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            # Return binary content (image data)
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {image_url}: {e}")
            raise
