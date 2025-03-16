from imread_from_url import imread_from_url

from dust3r import Dust3r, ModelType, get_device, calculate_img_size
from dust3r.visualization import visualize_output

# Model parameters
conf_threshold = 3.0
model_type = ModelType.DUSt3R_ViTLarge_BaseDecoder_512_dpt
device = get_device() # Cuda or mps if available, otherwise CPU
input_size = 224 if model_type == ModelType.DUSt3R_ViTLarge_BaseDecoder_224_linear else 512

# Read input images
frame1 = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/230128_Kamakura_Daibutsu_Japan01s3.jpg/800px-230128_Kamakura_Daibutsu_Japan01s3.jpg")
frame2 = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/The_Great_Buddha_of_Kamakura%2C_Kanagawa_Prefecture%3B_May_2011_%2806%29.jpg/960px-The_Great_Buddha_of_Kamakura%2C_Kanagawa_Prefecture%3B_May_2011_%2806%29.jpg")

# Calculate the new image size to keep the aspect ratio
width, height = calculate_img_size((frame1.shape[1], frame1.shape[0]), input_size)

# Initialize Dust3r model
dust3r = Dust3r(model_type, width, height, symmetric=True, device=device, conf_threshold=conf_threshold)

# Run Dust3r model
output1, output2 = dust3r(frame1, frame2)

# Visualize the output
visualize_output(output1, output2, visualize_depth=False)