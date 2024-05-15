import argparse
import logging

from diffusers import AutoPipelineForText2Image
import torch
from diffusers.utils import load_image
from utils import random_pad_pil_image, get_mask_image
from utils import random_pad_pil_image, get_mask_image, \
	replace_pixels, smooth_binary_image
import torch
from diffusers import AutoPipelineForInpainting
import random
from diffusers import StableVideoDiffusionPipeline
import torch
from diffusers.utils import load_image, export_to_gif


STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
INPAINTING_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
NEGATIVE_PROMPT = "bad object placements, deformed, disfigured, poor details,cluttered. blurred text. deformed textual labels"
BLANK_IMAGE_PROMPT = "plain ambient color background. no object."
BLANK_IMAGE_NEGATIVE_PROMPT = "black. white"
MASK_IMAGE_FILENAME = "mask_image.png"
DEBUG_ORIG_IMAGE_FILENAME = "orig_image.png"
DEBUG_TRANSFORMED_IMAGE_FILENAME = "transformed_image.png"
DEBUG_MASK_IMAGE_FILENAME = "mask_image.png"
DEBUG_BACKGROUND_IMAGE_FILENAME = "background_image.png"

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_blank_image():
	logger.info("Generating blank background image")
	pipeline = AutoPipelineForText2Image.from_pretrained(
		STABLE_DIFFUSION_MODEL, torch_dtype=torch.float16, variant="fp16"
	).to(device)
	
	image = pipeline(BLANK_IMAGE_PROMPT, negative_prompt=BLANK_IMAGE_NEGATIVE_PROMPT).images[0]
	return image

def image_inpainting(init_image, mask_image, prompt):
	logger.info("Performing image inpainting")
	pipeline = AutoPipelineForInpainting.from_pretrained(
		INPAINTING_MODEL, torch_dtype=torch.float16, variant="fp16"
	).to(device)
	

	image = pipeline(
		prompt=prompt, image=init_image, mask_image=mask_image, negative_prompt=NEGATIVE_PROMPT
	).images[0]
	return image


def generate_image(image_path, text_prompt, output_path, debug=False):
	logger.info(f"Generating image using provided image: {image_path}")
	logger.info(f"Text prompt: {text_prompt}")

	background_image = generate_blank_image()
	torch.cuda.empty_cache()

	init_image = load_image(image_path)
	img_width, img_height = init_image.size
	target_size = int(img_width * random.uniform(1.5, 3))


	background_image = background_image.resize((target_size, target_size))
	product_image = random_pad_pil_image(init_image, target_size=target_size)
	mask_image = get_mask_image(product_image)
	mask_image = smooth_binary_image(mask_image)
	mask_image.save(MASK_IMAGE_FILENAME)

	image = replace_pixels(background_image, product_image)
	

	if debug:
		logger.info("Saving debug images")
		init_image.save(DEBUG_ORIG_IMAGE_FILENAME)
		image.save(DEBUG_TRANSFORMED_IMAGE_FILENAME)
		mask_image.save(DEBUG_MASK_IMAGE_FILENAME)
		background_image.save(DEBUG_BACKGROUND_IMAGE_FILENAME)
		
	image = image_inpainting(image, mask_image, prompt=text_prompt)

	torch.cuda.empty_cache()
	

	return image
	


def create_video(image):
	# Load the pretrained pipeline
	pipeline = StableVideoDiffusionPipeline.from_pretrained(
	"stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
	).to(device)


	# Generate frames from the input image
	frames = pipeline(image, decode_chunk_size=8,).frames[0]
	torch.cuda.empty_cache()
	return frames


def main():
	parser = argparse.ArgumentParser(description="Image Generation Tool")
	parser.add_argument("--image", type=str, required=True, help="Path to the input image")
	parser.add_argument("--text-prompt", type=str, required=True, help="Text prompt for image generation")
	parser.add_argument("--output", type=str, required=True, help="Path to save the generated image")
	parser.add_argument("--debug", action="store_true",default=False)
	parser.add_argument("--video_output_path", type=str,default=None)

	args = parser.parse_args()

	image_path = args.image
	text_prompt = args.text_prompt
	output_path = args.output

	image = generate_image(image_path, text_prompt, output_path, debug=args.debug)
	
	
	logger.info(f"Saving generated image to: {output_path}")
	image.save(output_path)

	if args.video_output_path:
		frames = create_video(image)
		export_to_gif(frames, args.video_output_path)
	



if __name__ == "__main__":
	main()