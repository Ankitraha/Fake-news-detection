from readability import Document
import requests
import re
from readability import Document
import requests
import re
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from bs4 import BeautifulSoup
def link_input(link):
    url = link
    # Send a request to the URL and retrieve the page content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Create a Document object from the HTML content
        text = response.text
        cleaned_text = text.replace("&copy;", "")
        doc = Document(cleaned_text)

        # Extract the main text from the Document object
        main_text = doc.summary()
        # Clean up the main text by removing any unwanted characters
        main_text = re.sub(r'\n+', '\n', main_text)
        main_text = re.sub(r'\s+', ' ', main_text)
    
        # Remove any links present in the text
        main_text = re.sub(r'https?://[^\s]+', '', main_text)
        # Remove any texts that match the pattern "<p class="caption">...</p>"
        main_text = re.sub(r'<p class="caption">.*?</p>', '', main_text)
        # Remove any elements starting with "<" and ending with ">"
        main_text = re.sub("<.*?>", "", main_text)
        main_text_text = re.sub(r'\^ .*', '', main_text)


   


        # Extract the title of the web page
        title = doc.title()

        # Clean up the title by removing any unwanted characters
        title = re.sub(r'\n+', '\n', title)
        title = re.sub(r'\s+', ' ', title)
        

       # Print the main text and the title
        print("Title:", title)
        print("\nMain Text:\n", main_text)
    else:
        print("Failed to retrieve the page content")
        print("The input is a link:", link)
    x = [main_text]
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=maxlen) 
    prediction=(model.predict(x) >=0.5).astype(int)
    print("Prediction:", prediction)
     
def text_input(text):
    print("The input is text:", text)
    x = [text]
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=maxlen) 
    prediction=(model.predict(x) >=0.5).astype(int)
    print("Prediction:", prediction)
user_input = input("Enter text or link: ")

if user_input.startswith("http"):
    link_input(user_input)
else:
    text_input(user_input)
    
if user_input.startswith("http"):
    html_page = urllib.request.urlopen(user_input)
    soup = BeautifulSoup(html_page, 'html.parser')
    images = []
    for img in soup.find_all('img'):
        src = img.get('src')
        if src.endswith('.jpg'):
            src = "https:" + src
            images.append(src)

    for image_url in images:
        img_array = np.array(bytearray(urllib.request.urlopen(image_url).read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        plt.imshow(img)
        plt.show()
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

# Load the model
    model1 = load_model("keras_model.h5", compile=False)

# Load the labels
    class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
    real_image=0
    fake_image=0
    total_image=0
    for image_url in images:
        total_image+=1
        img_array = np.array(bytearray(urllib.request.urlopen(image_url).read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(img).convert("RGB")
        import PIL.Image
        if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
            PIL.Image.Resampling = PIL.Image

    # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
        image_array = np.asarray(image)

    # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
        data[0] = normalized_image_array

    # Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

    # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)
        if(index==0):
            real_image+=1
        else:
            fake_image+=1
    print("Total no. of images extracted:",total_image,"\n")
    print("No. of real images extracted:",real_image, "\n")
    print("No. of fake images extracted:",fake_image) 
