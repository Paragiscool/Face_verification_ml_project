import os
import urllib.request

# Ensure the directory exists
os.makedirs('application_data/verification_images', exist_ok=True)

# URLs of George W. Bush from the LFW dataset (hosted publicly)
bush_urls = [
    "http://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0001.jpg",
    "http://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0002.jpg",
    "http://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0003.jpg",
    "http://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0004.jpg",
    "http://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0005.jpg"
]

print("Downloading George W. Bush images...")
for i, url in enumerate(bush_urls):
    filename = f"application_data/verification_images/George_W_Bush_{i+1:04d}.jpg"
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

print("Done! You can now run the FaceID app.")
