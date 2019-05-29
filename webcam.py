# Author - Deepanshu Setia
# Date - 29 May'19
import cv2
import transformer
import torch
import utils
import sys

#STYLE_TRANSFORM_PATH = "trained_models/4.pth"
STYLE_TRANSFORM_PATH = "trained_models/"+str(sys.argv[1])+".pth"
PRESERVE_COLOR = False
WIDTH = 1280
HEIGHT = 720

def webcam(style_transform_path, width=1280, height=720):
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    print("Loading Transformer Network")
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_transform_path,map_location=device))
    net = net.to(device)
    print("Done Loading Transformer Network")

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)

    # Main loop
    with torch.no_grad():
        count = 1
        while True:
            # Get webcam input
            ret_val, img = cam.read()

            # AI Mirror 
            img = cv2.flip(img, 1)

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()
            
            # Generate image
            content_tensor = utils.itot(img).to(device)
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if (PRESERVE_COLOR):
                generated_image = utils.transfer_color(content_image, generated_image)
            img2 = cv2.imdecode(cv2.imencode(".png", generated_image)[1], cv2.IMREAD_UNCHANGED)

            count += 2
            # Show webcam
            cv2.imshow('Demo webcam', img2)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        
    # Free-up memories
    cam.release()
    cv2.destroyAllWindows()

webcam(STYLE_TRANSFORM_PATH, WIDTH, HEIGHT)
