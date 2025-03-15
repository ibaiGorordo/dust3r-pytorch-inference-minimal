import rerun as rr
from imread_from_url import imread_from_url

from dust3r import Dust3r, get_device, calculate_img_size

device = get_device() # Cuda or mps if available, otherwise CPU

# Read input images
frame1 = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/230128_Kamakura_Daibutsu_Japan01s3.jpg/800px-230128_Kamakura_Daibutsu_Japan01s3.jpg")
frame2 = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/The_Great_Buddha_of_Kamakura%2C_Kanagawa_Prefecture%3B_May_2011_%2806%29.jpg/960px-The_Great_Buddha_of_Kamakura%2C_Kanagawa_Prefecture%3B_May_2011_%2806%29.jpg")

# Initialize Dust3r model
conf_threshold = 3.0
width, height = calculate_img_size((frame1.shape[1], frame1.shape[0]), 512)
print(width, height)
model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
dust3r = Dust3r(model_name, width=width, height=height, symmetric=True, device=device, conf_threshold=conf_threshold)

output1, output2 = dust3r(frame1, frame2)

rr.init("Dust3r Visualizer", spawn=True)
rr.log("pts3d1", rr.Points3D(output1.pts3d))
rr.log("pts3d2", rr.Points3D(output2.pts3d))