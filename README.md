 Deep-Learning-Projects-Image-Classification-Electricity-Forecasting
 Deep Learning for Image Classification & Time-Series Forecasting
Two projects leveraging CNNs and RNNs for Flickr image categorization and electricity consumption prediction

 Project 1: Flickr Image Classification
Objective: Build a CNN-based model to classify Flickr images into 5 categories (animals, objects, people, scenes, others).

Key Steps & Results
Data Preprocessing:

Resized images to (255, 255, 3), normalized pixel values [0, 1].

Augmented data (rotation, flipping, zooming) to reduce overfitting.

Model Development:

MLP Baseline: Achieved ~50% accuracy (benchmark for comparison).

Optimized CNN: Reduced overfitting with dropout layers, reached 54.5% validation accuracy.

Transfer Learning: Fine-tuned VGG16 (frozen layers + custom head) for improved generalization.

Interpretability: Used Class Activation Maps (CAM) and Saliency Maps to visualize model decisions.

Tools: TensorFlow/Keras, OpenCV (image preprocessing).

 Project 2: Electricity Consumption Forecasting
Objective: Predict hourly electricity consumption in Romania using RNNs/CNNs.

Key Steps & Results
Dataset: Hourly consumption/production data (Kaggle) with 7 energy sources (nuclear, solar, etc.).

Models Compared:

LSTM + Attention: Test MAE = 0.0234 (best for long-term trends).

1D-CNN: Test MAE = 0.0217 (efficient for short-term patterns).

Insights:

LSTMs excelled in capturing temporal dependencies (>24h).

CNNs were faster but required dilated convolutions for longer sequences.

Tools: PyTorch, Pandas (time-series processing).

 Repository Structure
├── flickr_image_classification/  
│   ├── data/                # Preprocessed images (samples)  
│   ├── models/              # MLP, CNN, and VGG16 implementations  
│   ├── notebooks/           # Training/evaluation scripts  
│   └── results/             # Accuracy plots, CAM visualizations  
├── electricity_forecasting/  
│   ├── data/                # Processed CSV files  
│   ├── lstm_attention/      # LSTM + Attention code  
│   ├── cnn/                 # 1D-CNN implementation  
│   └── results/             # MAE comparisons, attention heatmaps  
├── docs/                    # Project report (PDF)  
├── Deep_Learning_GROUP4.pptx  # Presentation slides  
└── README.md               # This file  

 Key Takeaways
CNN Optimization: Data augmentation and transfer learning boosted image classification performance.

RNN vs. CNN: Choice depends on sequence length—LSTMs for long-term, CNNs for short-term patterns.

Business Impact:

Flickr: Better image categorization improves user searchability.

Energy: Accurate forecasts aid grid management and cost reduction.

 #DeepLearning #CNN #LSTM #TimeSeries #ComputerVision
