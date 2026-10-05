<h2>TensorFlow-FlexUNet-Image-Segmentation-OAI-Knee-Osteoarthritis-Edge-Detection (2026/10/05)</h2>
<h3>
OAI-Knee-Osteoarthritis-Edge-Detection: AI Generated Pseudo Masks Segmentation Challenge
</h3>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment in Image Segmentation for 
<a href="https://nda.nih.gov/oai"><b>The Osteoarthritis Initiative(OAI)</b></a> <b>Knee Osteoarthritis Edge Detection Two Classes</b>
 based on
our <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">TensorFlowFlexUNet Model</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and a 512x512-pixel upscaled PNG
 <a href="https://drive.google.com/file/d/1nUe7HsfisT7R2tEDoXymYnPZIQekZDHX/view?usp=sharing">
OAI-Knee-Osteoarthritis-Edge-Detection-ImageMask-Dataset.zip</a> with colorized masks, 
which was derived by us from the following dataset: 
<br><br>
<a href="https://www.kaggle.com/datasets/chauvvan/the-osteoarthritis-initiativeoai">
<b>The Osteoarthritis Initiative(OAI)</b>
</a>  by CITIVAN.
<br>
<br>
In this experiment, we aggregated the data originally categorized into four classes 
(Doubtful, Mild, Moderate, and Severe) into two classes (<b>Doubtful_or_Mild</b> and <b>Moderate_or_Severe</b>) 
for simplicity.
<br>
<br>
<hr>
<b>Actual Image Segmentation for OAI Knee Osteoarthritis Images of 512x512 pixels</b><br>
As shown below, the inferred masks resemble the ground-truth masks. <br>
<br>
<b>class_color_map = {Doubtful_or_Mild: green, Moderate_or_Severe: dark_red)} </b><br><br>
<table>
<tr>
<th width="320" height="auto">Input: image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Doubtful_9027189R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Doubtful_9027189R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Doubtful_9027189R.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Moderate_9053047R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Moderate_9053047R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Moderate_9053047R.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Moderate_9402139L.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Moderate_9402139L.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Moderate_9402139L.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from the following two datasets on the Kaggle website.
<br><br>
<a href="https://www.kaggle.com/datasets/chauvvan/the-osteoarthritis-initiativeoai">
<b>The Osteoarthritis Initiative(OAI)</b>
</a>
<br>by CITIVAN.
<br><br>
For more information, please refer to 
<a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/KneeOsteoarthritis.md">
Knee Osteoarthritis Dataset with Severity Grading</a>.
<br><br>
The following explanation (excerpt) was taken from the website above.
<br><br>
<b>Dataset Information</b><br>
This article introduces a dataset containing knee joint X-ray data used for knee joint detection and grading according 
to the Kellgren–Lawrence (KL) grading system. The dataset comprises 9,786 knee joint images categorized into five 
severity levels based on the KL system: 0 (healthy), 1 (doubtful), 2 (mild), 3 (moderate), and 4 (severe). 
All images have a resolution of 224 × 224 pixels. Approximately 40% of the dataset images belong to the healthy category, 18% are classified as doubtful, 26% as mild, 13% as moderate, and slightly over 3% as severe.
<br><br>
Knee Osteoarthritis (KOA) is one of the most common diseases among older adults, caused by the wearing down of 
the articular cartilage in knee joints. The accuracy of severity diagnosis significantly depends on the clinician's 
diligence and experience. The low reliability of clinicians' grading is attributed to the very subtle differences 
between X-ray images of adjacent grades. Detection and diagnosis of KOA is one of the fields where Deep Learning (DL) 
technology is applied. After training, data is fed into models that predict the severity of KOA based on the KL 
grading system. The high prevalence of KOA necessitates an accurate, reliable, 
and automated severity classification system, and deep learning offers one such solution.
<br><br>
<b>Citation</b><br>
<pre>
Chen, Pingjun (2018), “Knee Osteoarthritis Severity Grading Dataset”, Mendeley Data, V1, 
doi: 10.17632/56rmx5bjcr.1
</pre>
<br>
<b>License</b><br>
Unknown
<br>
<br>
<h3>
2. ImageMask-Dataset
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this <b>OAI Knee Osteoarthritis Edge Detection</b> Segmentation model,
 please download the dataset from Google Drive  
 <a href="https://drive.google.com/file/d/1nUe7HsfisT7R2tEDoXymYnPZIQekZDHX/view?usp=sharing">
OAI-Knee-Osteoarthritis-Edge-Detection-ImageMask-Dataset</a>. 
Expand the downloaded ImageMaskDataset and put it under the <b>./dataset</b> folder.
<br>
<pre>
./dataset
└─OAI-Knee-Edge-Detection
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
         ├─images
         └─masks
</pre>
<br>
<b>OAI-Knee-Edge-Detection Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/OAI-Knee-Edge-Detection_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images in the training and valid datasets is not large enough to use for the
 training set of our segmentation model.
<br>
<h3>2.2 Derivation of ImageMask Dataset</h3>
The folder structure of our <b>OAI-Images</b> derived from the original dataet excluded <b>Normal</b> is as follows,
but it contains no annotation (mask) files.
<br>
<pre>
./OAI-Knee-Osteoarthritis
 └─Images
    │
    ├─Doubtful
    │   ├─9000622L.png
...
    │   └─9999878L.png
    │    
    ├─Mild
    │   ├─9000099R.png
...
    │   └─9999878R.png
    │
    ├─Moderate
    │   ├─9000099L.png
...
    │   └─9999510L.png
    │
    └─Severe
        ├─9012867R.png
...
        └─9997856L.png
</pre>
It consits of the four grades, Doubtful,Mild, Moderate and Severe image data.
<br><br>
<b>Step 1</b><br>
We generated a 512x512-pixel upscaled master image dataset from the original 244x244-pixel PNG files in subfolders of <b>Images</b>.
<br><br>
<b>Step 2</b><br>
We generated the pseudo masks corresponding to the master images by applying a segmentation (inference) 
method of a pretrained FlexUNet model 
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Knee-X-Ray-Two-Classes-Edge-Detection">
TensorFlow-FlexUNet-Image-Segmentation-Knee-X-Ray-Two-Classes-Edge-Detection
</a> to all master images.
<br><br>
<b>Step 3</b><br>
We generated <b>curated pseudo masks</b> from the pseudo masks generated in the previous step
by using a simple Python script to exclude the inappropriate pseudo masks.
<br><br>
<b>Step 4</b><br>
We finally generated a small <b>OAI-Knee-Osteoarthritis-Edge-Detection-ImageMask-Dataset</b> 
from all pairs of the master images and their corresponding curated pseudo masks. <br>
<br>
<h3>2.3 Train Sample Images and Masks</h3>
<b>Train_sample_images</b><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_sample_masks</b><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3. Train TensorFlowFlexUNet Model
</h3>
 We trained the OAI-Knee-Edge-Detection TensorFlowFlexUNet model using the following
<a href="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
This runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16 </b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
; You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.04
; Specfied a large dilation.
dilation       = (3,3)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_focal_dice_loss" and <a href="./src/dice_coef_multiclass.py">"dice_coef_hybrid"</a>,
and weight parameters <b>hybrid_alpha</b> and <b>hybrid_beta</b> for <b>dice_coef_hybrid</b> function. 
<pre>
[model]
loss           = "categorical_focal_dice_loss"
metrics        = ["dice_coef_hybrid"]
; Experimental two weight parameters to calculate "dice_coef_hybrid" metric.
hybrid_alpha   = 1.6
hybrid_beta    = 0.4
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled the learning_rate_reducer callback and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with the patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specified RGB color map dict for OAI-Knee-Edge-Detection 1+2 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;OAI-Knee-Edge-Detection RGB color map dict for 1+2 classes.
rgb_map = {(0,0,0):0,(0,255,0):1,(180,20,20):2}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch change, the inference procedure can be called
 for 6 images in the <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1, 2, 3)</b><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 31, 32, 33)</b><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 64, 65, 66)</b><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 66 by EarlyStoppingCallback.<br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/train_console_output_at_epoch66.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4. Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection</b> folder,
and run the following bat file to evaluate the TensorFlowUNet model for OAI-Knee-Edge-Detection.<br>
<pre>
>./2.evaluate.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/evaluate_console_output_at_epoch66.png" width="1024" height="auto">
<br><br>Image-Segmentation-OAI-Knee-Edge-Detection
<a href="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_focal_dice_loss) on this OAI-Knee-Edge-Detection/test was low, but 
dice_coef_hybrid was not high, as shown below.
<br>
<pre>
categorical_focal_dice_loss,0.0245
dice_coef_hybrid,0.8694
</pre>
<br>
<h3>
5. Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection</b> folder
and run the following bat file to infer segmentation regions for images using the trained TensorFlowUNet model for OAI-Knee-Edge-Detection.<br>
<pre>
>./3.infer.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for OAI Knee Osteoarthritis Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks.<br>
<br>
<b>class_color_map = {Doubtful_or_Mild: green, Moderate_or_Severe: dark_red)} </b><br><br>
<table>
<tr>
<th width="320" height="auto">Image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Doubtful_9017876L.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Doubtful_9017876L.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Doubtful_9017876L.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Doubtful_9107048R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Doubtful_9107048R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Doubtful_9107048R.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Doubtful_9551114R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Doubtful_9551114R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Doubtful_9551114R.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Moderate_9053047R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Moderate_9053047R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Moderate_9053047R.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Moderate_9276684L.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Moderate_9276684L.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Moderate_9276684L.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/images/Severe_9326657R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test/masks/Severe_9326657R.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/OAI-Knee-Edge-Detection/mini_test_output/Severe_9326657R.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. THE OSTEOARTHRITIS INITIATIVE</b><br>
PROTOCOL FOR THE COHORT STUDY<br>
Michael C. Nevitt, PhD; David T. Felson, MD; Gayle Lester, PhD <br>
<a href="https://nda.nih.gov/static/docs/StudyDesignProtocolAndAppendices.pdf">
https://nda.nih.gov/static/docs/StudyDesignProtocolAndAppendices.pdf
</a>
<br><br>
<b>2. The 4 Stages of Knee Arthritis: What Your Grade Means (With X-Rays)</b><br>
Dr. Cory Calendine, MD<br>
<a href="https://corycalendinemd.com/blog/knee-arthritis-x-ray-grades/">
https://corycalendinemd.com/blog/knee-arthritis-x-ray-grades/
</a>
<br><br>
<b>3. Automatic knee osteoarthritis severity grading based on X-ray images using a hierarchical classification method</b><br>
Jian Pan, Yuangang Wu, Zhenchao Tang, Kaibo Sun, Mingyang Li, Jiayu Sun, Jiangang Liu, Jie Tian, Bin Shen 
<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11571664/">https://pmc.ncbi.nlm.nih.gov/articles/PMC11571664/</a>
<br><br>
<b>4. Ensemble deep-learning networks for automated osteoarthritis grading in knee X-ray images</b><br>
Sun-Woo Pi, Byoung-Dai Lee, Mu Sook Lee & Hae Jeong Lee <br>
<a href="https://www.nature.com/articles/s41598-023-50210-4">https://www.nature.com/articles/s41598-023-50210-4</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-OAI-Knee-Edge-Detection-Two-Classes-Edge-Detection </b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Knee-X-Ray-Two-Classes-Edge-Detection">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Knee-X-Ray-Two-Classes-Edge-Detection
</a>
<br><br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-OAI-Knee-Edge-Detection-Edge-Detection </b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Knee-X-Ray-Edge-Detection">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Knee-X-Ray-Edge-Detection
</a>
<br><br>
<b>7. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br><br>
