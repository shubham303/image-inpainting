
import cv2
import numpy as np
from PIL import Image , ImageFilter
import random
import cv2
import numpy as np


def random_pad_pil_image_with_edge(img, target_size=2048, img_size=512):
    """
    Resizes the input PIL image and its Canny edge-detected version to 128x128, and then pads them randomly to the target size (512x512) with white pixels.
    
    Args:
        img (PIL.Image.Image): Input PIL image.
        target_size (int): Target size of the padded images. Default is 512.
    
    Returns:
        Tuple[PIL.Image.Image, PIL.Image.Image]: Padded input image and padded Canny edge-detected image, both with size (target_size, target_size).
    """
    # Resize the input image to 128x128
    #img = img.resize((img_size, img_size))
    
    # Convert the PIL image to a NumPy array
    img_array = np.array(img)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 100, 200)
    
    # Convert the NumPy array back to a PIL image
    edges_img = Image.fromarray(edges)
    
    # Create a new image with the target size and white background
    padded_img = Image.new('RGB', (target_size, target_size), (254, 254, 254))
    padded_edges_img = Image.new('L', (target_size, target_size), 0)
    
    # Get the width and height of the resized images
    w, h = img.size
    
    import random
    pad_top = random.randint(0, target_size - h)
    pad_left = random.randint(0, target_size - w)
    
    # Paste the resized images onto the new images
    padded_img.paste(img, (pad_left, pad_top))
    padded_edges_img.paste(edges_img, (pad_left, pad_top))
    
    return padded_img, padded_edges_img

def random_pad_pil_image(img, target_size=2048):
    """
    Resizes the input PIL image and its Canny edge-detected version to 128x128, and then pads them randomly to the target size (512x512) with white pixels.
    
    Args:
        img (PIL.Image.Image): Input PIL image.
        target_size (int): Target size of the padded images. Default is 512.
    
    Returns:
        Tuple[PIL.Image.Image, PIL.Image.Image]: Padded input image and padded Canny edge-detected image, both with size (target_size, target_size).
    """
    # Resize the input image to 128x128
    #img = img.resize((img_size, img_size))
    
    # Convert the PIL image to a NumPy array
    img_array = np.array(img)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 100, 200)
    
    # Convert the NumPy array back to a PIL image
    edges_img = Image.fromarray(edges)
    
    # Create a new image with the target size and white background
    padded_img = Image.new('RGB', (target_size, target_size), (254, 254, 254))
    
    
    # Get the width and height of the resized images
    w, h = img.size
    
    import random
    pad_top = random.randint(0, target_size - h)
    pad_left = random.randint(0, target_size - w)
    
    # Paste the resized images onto the new images
    padded_img.paste(img, (pad_left, pad_top))
    
    return padded_img




def get_mask_image(input_image):
    """
    Takes a PIL image as input and returns a mask image.
    The mask image will have a value of 255 wherever the input image pixel has a value of 254,
    and 0 elsewhere.
    
    Args:
        input_image (PIL.Image.Image): The input PIL image.
        
    Returns:
        PIL.Image.Image: The mask image.
    """
    # Convert the input image to grayscale
    input_image = input_image.convert('L')
    
    # Get the size of the input image
    width, height = input_image.size
    
    # Create a new blank image for the mask
    mask = Image.new('L', (width, height), color=0)
    
    # Get the pixel data of the input image
    input_pixels = input_image.load()
    
    # Get the pixel data of the mask image for writing
    mask_pixels = mask.load()
    
    # Iterate over each pixel in the input image
    for x in range(width):
        for y in range(height):
            # If the pixel value is 254, set the mask pixel to 255, else 0
            if input_pixels[x, y] > 250:
                mask_pixels[x, y] = 255
            else:
                mask_pixels[x, y] = 0
    
    return mask



def make_canny_condition(image):

    image = np.array(image)

    image = cv2.Canny(image, 100, 200)

    image = image[:, :, None]

    image = np.concatenate([image, image, image], axis=2)

    image = Image.fromarray(image)

    return image




def merge_images(image1, image2):
    # Get the dimensions of the first image
    width1, height1 = image1.size
    
    # Get the dimensions of the second image
    width2, height2 = image2.size
    
    # Create a new image with the same dimensions as the first image
    merged_image = Image.new('RGB', (width1, height1))
    
    # Paste the first image onto the new image
    merged_image.paste(image1, (0, 0))
    
    # Generate random coordinates for the top-left corner of the second image
    x = random.randint(0, width1 - width2)
    y = random.randint(0, height1 - height2)
    
    # Paste the second image onto the new image at the random coordinates
    merged_image.paste(image2, (x, y))
    
    return merged_image




def replace_pixels(img1, img2):
    # Convert images to RGB mode if they are not already
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')
    
    # Get the dimensions of the images
    width, height = img1.size
    
    # Create a new image with the same dimensions as the input images
    new_image = Image.new('RGB', (width, height))
    
    # Convert the second image to grayscale
    img2_grayscale = img2.convert('L')
    
    # Iterate over the pixels of the images
    for x in range(width):
        for y in range(height):
            # Get the pixel values from the input images
            pixel1 = img1.getpixel((x, y))
            pixel2 = img2.getpixel((x, y))
            
            # Get the grayscale value of the corresponding pixel in the second image
            grayscale_value = img2_grayscale.getpixel((x, y))
            
            # Replace the pixel in the new image with the pixel from img2
            # if the grayscale value is less than 250, else keep the pixel from img1
            if grayscale_value < 250:
                new_image.putpixel((x, y), pixel2)
            else:
                new_image.putpixel((x, y), pixel1)
    
    return new_image




def smooth_binary_image(image, kernel_size=5, sigma=1):
    """
    Smooths a binary PIL image with values 0 or 255.
    
    Args:
        image (PIL.Image.Image): The input binary image.
        kernel_size (int, optional): The size of the Gaussian kernel. Defaults to 5.
        sigma (float, optional): The standard deviation of the Gaussian kernel. Defaults to 1.
        
    Returns:
        PIL.Image.Image: The smoothed binary image.
    """
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Apply Gaussian blur
    blurred = Image.fromarray(image_array.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=sigma))
    blurred_array = np.array(blurred)
    
    # Binarize the blurred image
    smoothed_array = np.where(blurred_array > 127, 255, 0).astype(np.uint8)
    
    # Convert the smoothed array back to a PIL Image
    smoothed_image = Image.fromarray(smoothed_array)
    
    return smoothed_image