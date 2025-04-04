import torch
import torchvision.models.video as models
import torchvision.transforms as transforms
#from transformers import I3DModel, I3DProcessor
from torchvision.io import read_video
#from torchvision.transforms import UniformTemporalSubsample
import torch.nn.functional as F
import av
from sentence_transformers import SentenceTransformer, util
import cv2
import numpy as np
import os
import torch
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#from gensim.models import Doc2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as T
from torchvision.models.video import r3d_18
import cv2
import os
from pytorchvideo.models.hub import i3d_r50
from PIL import Image
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import yt_dlp

def download_video(url, output_path, name):
    ydl_opts = {
        'outtmpl': f"{output_path}/{name}",
        'cookies-from-browser': 'chrome',  # Change to 'firefox' if needed
        'format': 'best',  # Get the best available quality
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Video downloaded successfully: {output_path}")
        return os.path.join(output_path, name)
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None
    
def get_MSR_VTT_data(output_dir_vids, download = 0, subset = 20): #dataset of youtube vids, single summary
    dataset = load_dataset("AlexZigma/msr-vtt", split="train")
    print(dataset[0])  
    subset = dataset.select(range(subset)) 
    paths = []
    captions =[]
    i=0
    video_paths = ''
    for entry in subset:
        video_url = entry['url']  
        caption = entry['caption'] 
        name=f"vid_{i}.mp4"
        if download==1:
            video_paths = download_video(video_url,output_dir_vids, name) 
        else:
            video_paths = os.path.join(output_dir_vids, name)

        if video_paths is not None:
            print(f"video path {video_paths}")
            paths.append(video_paths)
            captions.append(caption)
            i+=1
    return paths, captions


def get_ActivityNet_data(output_dir_vids, download = 0, subset = 10):
    dataset = load_dataset("activitynet", split="test")
    print(dataset[0])  
    subset = dataset.select(range(50))
    paths = []
    captions =[]
    i=0
    for entry in subset:
        video_url = entry['url']  
        caption = entry['caption'] 
        name=f"vid_{i}.mp4"
        if download:
            video_path = download_video(video_url,output_dir_vids, name) 
        else:
            video_path = os.path.join(output_dir_vids, name)

        if video_path is not None:
            print(f"video path {video_path}")
            paths.append(video_path)
            captions.append(caption)
            i+=1
    return paths, captions

def create_i3d():
    model = i3d_r50(pretrained=True)
    model.eval()
    return model

'''def create_i3d_fb():
    processor = I3DProcessor.from_pretrained("facebook/i3d-r50")
    model = I3DModel.from_pretrained("facebook/i3d-r50")
    model.eval()
    return model, processor'''

def create_r3d():
    model = r3d_18(pretrained=True)  
    model.eval()
    return model

def create_mvit_v2():
    model = models.mvit_v2_s(weights="KINETICS400_V1")
    model.eval()
    return model

def create_blip2vid():
    '''processor = BlipProcessor.from_pretrained("kpyu/video-blip-opt-2.7b-ego4d")
    model = BlipForConditionalGeneration.from_pretrained("kpyu/video-blip-opt-2.7b-ego4d")'''
    #Salesforce/blip2-flan-t5-xl
    processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", force_download=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", force_download=True)

    model.eval()
    return model, processor

def create_blip2():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return model, processor

def create_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    gpt_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, gpt_model


def load_kinetics_classes(): 
    import requests
    url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
    response = requests.get(url)
    class_names = response.text.strip().split("\n")
    class_names = [line.split(", ") for line in class_names]
    return class_names


def create_transform():
    transform = transforms.Compose([ #to make sure its the correct size
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return transform

def vid_preprocess_clip(video_path, output_dir, model, processor, batch = 1, seq_frames=16):
    video = cv2.VideoCapture(video_path) #video capture from path
    frames = []
    frames_batch = []
    idv_actions =[]
    frame_counter = 0
    if not video.isOpened():
        print(f"Failed to open video file: {video_path}")
        return
    print(f"Successfully opened video file: {video_path}")

    while video.isOpened(): #keep open for all frames
        ret, frame = video.read()
        if not ret: #if it couldnt read the frame
            break
       
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #change from bgr to rgb color

        if processor == None:
            transform = create_transform()
            frame = transform(frame) #transforms using the above trans func
        else:
            frame =  processor(frame, return_tensors="pt", padding=True, truncation=True)

        frames.append(frame) #collect frames

        if len(frames) == seq_frames: #when enough frames for a clip
            frame_tensor = torch.stack(frames) #.unsqueeze(0) #unsqueeze adds batch dim after stacking tensors
            video_tensor = torch.stack(frames).unsqueeze(0).permute(0, 2, 1, 3, 4)
            if not batch:
                actions = clip_process_indv(video_tensor, model)
                idv_actions.append(actions)
            else: #for batch
                frames_batch.append(frame_tensor)

            frames = [] #reset frames

        frame_counter+=1
    video.release()
    
    if not batch:
        return idv_actions
    if batch:
        video_tensor_batch = torch.stack(frames_batch).permute(0, 2, 1, 3, 4)# reorders to (batch, channels, frames, h, w)
        print(f"size {video_tensor_batch.shape}")
        return video_tensor_batch
    


def frame_process_idv(frame, model, processor):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #otherwise the colors get confused
    inputs = processor(images=frame_rgb, return_tensors="pt")
    with torch.no_grad(): #dont want to waste space and time on gradients
        outputs = model.generate(**inputs) #gets descriptions
        caption = processor.batch_decode(outputs, skip_special_tokens=True)[0] #converts to readable text
    return caption


def vid_preprocess_image(video_path, output_dir, processor, model, batch = 1, seq_frames=16):
    video = cv2.VideoCapture(video_path) #video capture from path
    frames_batch = []
    captions = []
    frame_counter= 0
    if not video.isOpened():
        print(f"Failed to open video file: {video_path}")
        return
    print(f"Successfully opened video file: {video_path}")

    while video.isOpened(): #keep open for all frames
        ret, frame = video.read()
        if not ret: #if it couldnt read the frame
            break

        if frame_counter % seq_frames == 0: #1 frame captured per seq len
            frame_path = os.path.join(output_dir, f"frame_{frame_counter}.jpg")
            cv2.imwrite(frame_path, frame) #writes frame to path
            if not batch:
                caption = frame_process_idv(frame, model, processor)
                captions.append(caption)
            else:
                frames_batch.append(frame)

        frame_counter+=1
    video.release()
    if not batch:
        return captions
    else:
        return frames_batch
    


def frame_process_batch(frames, model, processor):
    embeddings = []
    tokenizer, gpt_model = create_gpt2()
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=frame_rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs) #gets descriptions
            caption = processor.batch_decode(outputs, skip_special_tokens=True)[0] #converts to readable text
            
            inputs = tokenizer(caption, return_tensors="pt")
            embedding = gpt_model.transformer.wte(inputs.input_ids).mean(dim=1)
            embeddings.append(embedding)
        if not embeddings:
            raise RuntimeError("No embeddings generated.")
    return torch.cat(embeddings) if embeddings else torch.tensor([])


def clip_process_indv(video_tensor, model): #returns an action and probability
    class_names = load_kinetics_classes()
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = probabilities.topk(1)
    
    actions = [f"{class_names[cls.item()]} ({prob:.2f})" for cls, prob in zip(top_classes[0], top_probs[0])]
    return actions



def clip_process_batch(video_tensor, model):
    with torch.no_grad(): 
        outputs = model(video_tensor) 
        probabilities = torch.nn.functional.softmax(outputs, dim=1) #probabililties, dim 1 is classes
        top_probability, top_classes = probabilities.topk(10) #top 10
    return top_probability, top_classes

#init clip class
def clip_process_init(video_path, output_dir, model, processor, batch=1, seq_frames=16):
    class_names = load_kinetics_classes()
    video_tensor= vid_preprocess_clip(video_path, output_dir, model, processor, batch = batch, seq_frames = seq_frames ) 
    if video_tensor is None:
        return "Failed to process video."
    if batch:
        probabilities, classes = clip_process_batch(video_tensor, model) #gets the class indices and probabilities
        #cls is a pytorch obj, classes[] is our tensor of class indices, probabilities[] is our tensor list of probabilities
        actions = [f"{class_names[cls.item()]}: {probability:.4f}" for cls, probability in zip(classes[0], probabilities[0])]
        actions = [action.replace("['", "").replace("']", "") for action in actions]
        
        return "The video shows: " + ", ".join(actions) + "."
    else:
        print(f"Clip review indv: {video_tensor}")
        return None


#init images
def frame_process_init(video_path, output_dir, model, processor, batch = 1, seq_frames = 16):
    video_tensor = vid_preprocess_image(video_path, output_dir, model, processor, batch = batch, seq_frames = seq_frames ) 
    if video_tensor is None:
        return "Failed to process video."
    if batch:
        captions_batch = frame_process_batch(video_tensor, model, processor)
        print(f"Captions Batch Image: {captions_batch}")
        return captions_batch
    else:
        print(f"Image Captions Idv: {video_tensor}")


def calculate_similarity_cosine(predicted_label, actual_caption): #word matches essentially
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([predicted_label, actual_caption])
    svd = TruncatedSVD(n_components=2)
    X_lsa = svd.fit_transform(X)
    similarity_score = cosine_similarity(X_lsa[0:1], X_lsa[1:2])[0][0]
    return similarity_score


def calculate_similarity_transformer(predicted_label, actual_caption): #captures context, semantic
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  
    embedding1 = model.encode(predicted_label, convert_to_tensor=True)
    embedding2 = model.encode(actual_caption, convert_to_tensor=True)
    similarity_score = util.cos_sim(embedding1, embedding2)
    print(f"Similarity Score: {similarity_score.item():.4f}")
    return similarity_score.item()

def i3d_and_r3d(model, video_path, segment = 1, overSample = 0):
    from torchvision.transforms.functional import to_pil_image
    from torchvision.transforms import InterpolationMode
    transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=InterpolationMode.BILINEAR),  # Ensure correct resizing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])
    if segment:
        container = av.open(video_path)
        video_duration = container.duration / av.time_base
        end_pt = min(240, video_duration) #whichever happens first
        start_pt = 10
        if start_pt >= end_pt:
            start_pt = 0
        #info = read_video(video_path, pts_unit="sec", start_pts=2, end_pts=end_pt, output_format="TCHW")[2]
        video, _, _ = read_video(video_path, pts_unit="sec", start_pts=2, end_pts=end_pt)# gets frames, Video shape: (T, H, W, C)
    else:
        video, _, _ = read_video(video_path, pts_unit="sec") 

    num_frames = 16
    if overSample:
        T = video.shape[0]  # Total frames in the extracted segment
        if T > num_frames:
            indices = torch.linspace(0, T - 1, steps=num_frames *2).long()
            print(f"first indices {indices}")
            indices = indices[::2] 
            video = video[indices] 
            print(f"indices {indices}")
        elif T < num_frames:
            # If there are fewer frames, duplicate some
            repeat_factor = (num_frames + T - 1) // T  # How many times to repeat
            video = video.repeat_interleave(repeat_factor, dim=0)[:num_frames] 
            print(f"repeat factor {repeat_factor}") 
            print(f"length {video.shape}")
    video = video.permute(0, 3, 1, 2)
    if video.shape[0] > num_frames:
        indices = torch.linspace(0, video.shape[0] - 1, steps=num_frames).long()
        video = video[indices] 
    video = torch.stack([transform(to_pil_image(frame)) for frame in video])  # Resize and normalize
    video = video.permute(1, 0, 2, 3).unsqueeze(0)

    output = model(video) 
    probs = F.softmax(output, dim=1) 
    top5 = torch.topk(probs, 5)
    class_labels = load_kinetics_classes()

    # Print predictions
    for i in range(5):
        label_idx = top5.indices[0][i].item()
        confidence = top5.values[0][i].item()
        #print(f"Prediction: {class_labels[label_idx]} | Confidence: {confidence:.4f}")
    label_idx = top5.indices[0][0].item()
    confidence = top5.values[0][0].item()
    return f"Prediction: {class_labels[label_idx]} | Confidence: {confidence:.4f}"


def sklearn_metrics(scores):
    pred_labels = [1 if score >= 0.2 else 0 for score in scores]
    true_labels = [1] * len(pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


current_dir = os.getcwd()
current_dir2 = os.path.join(current_dir, "Documents", "UNM2025", "Videos")
video_filename = "testVid2.mp4"
video_path = os.path.join(current_dir2, video_filename)
output_dir = os.path.join(current_dir2, "frames")
output_dir_vids = os.path.join(current_dir2, "vids")
os.makedirs(output_dir, exist_ok=True)
#video_path = os.path.join(output_dir_vids, "vid_0.mp4")

#videos, captions = get_MSR_VTT_data(output_dir_vids, download=1)
model = create_mvit_v2()
'''descriptions= []
for i, video in enumerate(videos):
    description = i3d_and_r3d(model, video)
    if description is not None and description != []:
        descriptions.append(description)
        print(f"Clip Review Batch: {description}")
        print(f"Actual Caption: {captions[i]}")

scores = []
predictions = [re.search(r"\['(.*?)'\]", pred).group(1) for pred in descriptions]
#how did we do?
for i, pred in enumerate(predictions):
    score = calculate_similarity_transformer(pred, captions[i])
    scores.append(score)


mean_similarity = np.mean(scores)
print(f"Average Similarity Score: {mean_similarity:.4f}")
sklearn_metrics(scores)

'''
#model, processor = create_blip2()
#frame_process_init(video_path, output_dir, model, processor, batch = 1, seq_frames = 16)
#test our vids
descriptions = []
captions = ["woman makes flower display", "person flips through book", "Dog walks down path with red flows, man follows", 
            "horse runs in dirt corral", "people walk in indian fish market", "person buys produce in market", 
            "person plays piano", "people in office work", "people ice skate in front of building", "man kicks soccar ball in red shirt", 
            "woman works in lab, uses a scanner"]
for i in range(1, 11):
    video_filename = f"{i:02d}.mp4"
    video_path = os.path.join(current_dir2, video_filename)

    if os.path.exists(video_path):
        description = i3d_and_r3d(model, video_path, segment =0)
        if description is not None and description != []:
            descriptions.append(description)
            print(f"Clip Review Batch: {description}")
            print(f"Actual Caption: {captions[i-1]}")
    else:
        print(f"Video file {video_filename} not found. Skipping...")
        continue

scores = []
predictions = [re.search(r"\['(.*?)'\]", pred).group(1) for pred in descriptions]
#how did we do?
for i, pred in enumerate(predictions):
    score = calculate_similarity_transformer(pred, captions[i])
    scores.append(score)


mean_similarity = np.mean(scores)
print(f"Average Similarity Score: {mean_similarity:.4f}")
sklearn_metrics(scores)



'''model =create_r3d()
i3d_and_r3d(model)
model = create_i3d()
i3d_and_r3d(model)
model, _= create_blip2vid()
i3d_and_r3d(model)'''


