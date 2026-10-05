# Gesture detection model

## What is needed:

A way to identify and classify body gestures.

The current gesture list is: 
- Shake hand
- High five
- Hug
- High wave
- Clap
- Left kiss
- Heart
- Right heart
- Hands up
- Right hand up
- Right kiss
- Two hand kiss 

Which were retrieve from the G1 pre-made gestures. 

## Approach

Gesture detection not only requires frame by frame analysis, but also a history of frames to analyze as there are several gestures that statically don't represent anything meaningful, but rather are composed by dynamic body parts. i.e wave, clap, etc. 

Given this constraint, there are several approaches I've come up with that can be interesting as an initial starting point:
- Finetuned YOLO Model. This is a very straightforward approach, where you use mediapipe's body detection to extract body coordiantes and generate bounding boxes based off that which can then be re-trained on a YOLO model to detect those gestures. However, this will make a visual analysis rather than use body parts to make a guess. Aditionally, this will only analyze each frame individually, rather than constructing a history of movements, so the full focus of the project won't be achieved. Here mediapipe would only be used to speed up the labeling process, and given YOLO's architecture, it doesn't react to body parts or concepts but rather pixels, so it won't work precisely. **I tried it and gave me mixed results**
- Embedding-based classifier using mediapipe detections and cosine similarity to quickly discover similarities with pre-recorded gestures. This allows for quick modifications to the gesture list as we would only need to add new embeddings rather than retrain a new model. This works by using landmark vectors and performing cosine similarity with a vector database where the embeddings for each vector are stored, which will help define which gesture is being detected. Similar to the previous approach, this will work well for static gestures, as a single frame will contain everything needed to classify a model, plus, as we will be using mediapipe's detection as a source of information, this will give a more accurate result rather than comparing training images as in the YOLO approach; here we are comparing body points relative to the gestures. **I tried it with a small datased and worked kinda nice for static gestures only** 
- Sequence model(GRU/LSTM) over mediapipe landmarks. Instead of looking at one input in isolation, we read the input one state at a time and keep a hidden state summary of what's been seen so far. This gives use the "memory" part that was not possible in the last two approaches. Using mediapipe, each step we would process the current mediapipe landmarks and use the summary to analyze if the movements correspond to any particular gesture. The neural network decides what to include in its summary and what to forget. Claude's recommednation for this was to use a GRU because it "has fewer parameters and trains faster, and it usually performs about the same, which makes it a good fit for small datasets", its also insensitive to clothing. While it probably will have better results, it does make training a bit more tedious than the previous approach, as the we would need to retrain the entire neural network :( **I am working on this one** 
- DTW (Dynamic Time Warping) nearest-neighbor over landmark sequences. No training: new live sequences are compared against stored reference clips, aligning them in time so differences in speed or clip length don't matter. Adding a gesture only requires recording reference clips. It is slower at inference and less robust than a trained model as the reference set grows. **Claude suggested this approach, I haven't tried it**

## Contributors 
- [@jpgtzg](https://github.com/jpgtzg)
