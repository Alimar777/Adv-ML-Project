import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return 1 (if one GPU)
print(torch.cuda.get_device_name(0))  # Should print "RTX 2060 Super"


from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
#image_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.internetconsultancy.pro%2Fwp-content%2Fuploads%2F2017%2F06%2Fesmee-holdijk-130494-e1496843169448.jpg&f=1&nofb=1&ipt=eef5bbad7c7d68d9531439f4248313bd61963bcbdc661d3b2f80be8c0befac24&ipo=images"  # Replace with an actual image URL
#image_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmakeawebsitehub.com%2Fwp-content%2Fuploads%2F2014%2F12%2F130.jpg&f=1&nofb=1&ipt=7b083dd70e30182f007007d5b5535eb746babdb594a49ee798e5c891c40a472a&ipo=images"
#image_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fqodeinteractive.com%2Fmagazine%2Fwp-content%2Fuploads%2F2019%2F08%2FFeatured-Stock.jpg&f=1&nofb=1&ipt=5f8010f651dddfe151f08ef849844d32f4b63f0640f4aa054f80421a245208e8&ipo=images"
#image_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcreateandcode.com%2Fwp-content%2Fuploads%2F2017%2F01%2Ffree-stock-photos.jpg&f=1&nofb=1&ipt=cfd785d6364b2feae250531f3828e597ed1531bc7e145396ebaf5ec435e840f4&ipo=images"
#image_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimages.freeimages.com%2Fimages%2Flarge-previews%2F48d%2Fmarguerite-1372118.jpg&f=1&nofb=1&ipt=c1eeacb1f003f9572147a6144301cb7921410c06cc15b36d9bc44586f6967606&ipo=images"
image_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.pointgadget.com%2Fwp-content%2Fuploads%2F2020%2F05%2Ffree-stock-images-websites.jpg&f=1&nofb=1&ipt=431f7865d92a7b6f956a213a3fc8dbdc069edba1568ecf56a641cbd9dc62cf21&ipo=images"

image = Image.open(requests.get(image_url, stream=True).raw)

# Generate caption
inputs = processor(images=image, return_tensors="pt", max_new_tokens=30)
caption_ids = model.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)

print("Caption:", caption)
