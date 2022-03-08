# CS4641_ML_Project

## Final Report Video
https://youtu.be/NrBoEyrouU8

## Introduction / Background
The topic of our project is related to diabetic retinopathy (DR) detection for which we are going to train a deep learning model that is able to identify signs of diabetic retinopathy in eye images. Diabetic retinopathy is the most common diabetic eye disease and a leading cause of blindness in American adults. The retina is the light-sensitive tissue at the back of the eye. A healthy retina is necessary for good vision. According to the US national eye institute, from 2010 to 2050, the number of Americans with diabetic retinopathy is expected to nearly double, from 7.7 million to 14.6 million. As the number of people being diagnosed with diabetic retinopathy increases, it is necessary to construct an automated image classification model to automatically classify retinal tissue into healthy and pathological in early stages.

## Problem Definition
While taking active care in diabetes management can prevent diabetic retinopathy, with more than 1 in 10 people in the United States with diabetes, it is no doubt that diabetic retinopathy is also a prevalent complication that needs attention (Mayo Clinic, 2021). With the advanced technology and the ampleness of patient/non-patient data, our goal in this project is to use classification models using real data, to predict and diagnose diabetic retinopathy in patients.

## Data Collection
We used data from the Diabetic Retinopathy Detection dataset from Kaggle. The dataset consisted of a total of 35,127 training images, which were JPEG files of left and right eyes. There were a total of five classes (No DR, Mild, Moderate, Severe, Proliferate DR) within the training set. Because of limited resources to train the deep learning model, we took a random sample from each class to avoid data imbalance. Specifically, the training set consists of 560 images from each class, which sums up to 2800 images in total. Similarly, for the validation set, we have 140 images from each class (700 images in total) and for the test set, we have 100 images from each class (500 images in total). In percentage terms, the breakdown is 70% training, 17.5% validation, and 12.5% testing. All images were 3-channel colored images but had differing sizes. 

## Methods
### Image preprocessing
Before feeding the data into the model, we cleaned up our image data by performing a series of image preprocessing steps. First, a cropping method was run to make the images a uniform size of 256x256 pixels. Below is an example of the pre-cropped image and the cropped image of the eye.

![1](https://github.gatech.edu/storage/user/54998/files/498b2c85-fd62-4a2f-a940-18d59b046b69)
![2](https://github.gatech.edu/storage/user/54998/files/af324908-1535-401a-8443-482b348ccac7)<br/>
***image above: original image (left) and cropped image (right).***

Additionally, a normalization function, ranging from 0 to 1, was run to adjust the lighting of the images. Below are images of the non-normalized data and the normalized data.

![3](https://github.gatech.edu/storage/user/54998/files/14bd3b74-60ae-4e86-afad-8412d6dc0f5e)
![4](https://github.gatech.edu/storage/user/54998/files/cd65d4e0-bcd4-45d5-bb5b-042211083564)<br/>
***image above: cropped image (left) and cropped AND normalized image (right).***

The ‘after normalization’ image visually looks completely black since the color values were adjusted from range [0, 255] to [0, 1] for each red, green, and blue values. The human eye would not be able to distinguish the differences; however, when the model reads the dataset, it would distinguish and read the scaled down values of the normalized data.

Below is a histogram of ‘16_right.jpeg’ containing the pixel values for the pixels in the image:<br/>

![5](https://github.gatech.edu/storage/user/54998/files/2d8abb6f-45e3-4074-8977-04885a487abe)
![6](https://github.gatech.edu/storage/user/54998/files/e021e87e-93b3-41c4-ab73-69f3d951eb9d)<br/>
***Figure 1. Histograms of pixel values in non-normalized image (left) and normalized image (right).***

One of the experiments was conducted to examine the effects of image preprocessing, specifically focusing on normalization of images. After analyzing the results from the midterm project, we realized that the normalization logic we followed was flawed. Figure 2a shows the histogram of pixel values of the original image. Figure 2b shows the histogram of pixel values of the same image that was normalized but saved as a JPEG file afterwards. Figure 2c shows the corrected normalization method, which shows the same distribution as the original image, but distributed between 0 and 1. We suspected that the different distribution shape as shown in Figure 2b was caused when we were converting the image to a JPEG file. Therefore, we eliminated this problem by instead of saving it as JPEG files, normalizing the image right before it entered the deep learning model. This corrected normalization method is reflected in Table 1 under “Effects of normalizations”. The flawed normalization method and its results are not reflected in this paper.

![7](https://github.gatech.edu/storage/user/54998/files/e01e8984-999f-4b48-a7b1-690484fc6d53)
![8](https://github.gatech.edu/storage/user/54998/files/19e95735-472c-48bf-973b-037514ee7291)
![9](https://github.gatech.edu/storage/user/54998/files/c97488e4-d9ae-441f-bb7e-796394f8c851)<br/>
***Figure 2. Histogram of pixel values for orginal image(2a), using wrong normalization logic(2b) and using correct normalization logic(2c)***

![10](https://github.gatech.edu/storage/user/54998/files/5a6f52ff-d8d7-46bb-bbcb-ab850c2908f1)<br/>
***Table 1. Table of comprehensive experiments.***
*Every parameter and results indicating normalized datasets indicate the corrected normalization methods.*

### Deep learning model - supervised approach
**Model initialization**: A modified VGG-16/ResNet18 model pre-trained on ImageNet was used for our 5-category classification in a transfer learning approach. Since the model was pre-trained on more than a million images from the ImageNet database, we believe that it carries strong ability to extract generic features from the dataset. To save computation time, we froze all weights except for the last layer in the model. To modify and adapt the pretrained VGG-16/ResNet18 model to our diabetic retinopathy detection project, we first printed out and examined the architecture of the model using the .eval() method, then substituted the last fully-connected layer and changed the out_features variable from the original 1,000 to 5 to achieve a five-category output. We have noticed that VGG16 and ResNet 18 have different layouts in terms of architecture, so we have modified our code accordingly.

**Generic model class**: To lay out the pipeline more clearly, we have created a generic model class that takes in the modified model, the train DataLoader, the test DataLoader, the criterion (loss function), and the optimizer, and outputs the running loss and the accuracy. 

**Model training**: After we initialized and modified the VGG16 model, we determined the criterion and optimizer that would be used during our model training process. We used cross entropy as the loss function and stochastic gradient descent (SGD) as the optimizer for our model. The optimizer was set to have only the parameters of the classifiers being optimized. We trained the model using the following hyperparameters: training lasted for 100 epochs with batch size = 56, learning rate = 0.001, momentum = 0.9, and weight decay = 0.0005.

### Performance metrics
**Accuracy**: the first metric that comes to our mind is the accuracy score, which is the percentage of predicted labels that match the true labels exactly. To calculate this metric, we used the accuracy_score() function from the sklearn.metrics package.

**Confusion matrix**: further, since our model is a multi-class classification model, we also looked at the confusion matrix to see how well the model performs on the test data. This gives us more information than a single accuracy score: if the predicted labels were wrong, we could see which classes the images were misclassified into. We used the confusion_matrix() function from the sklearn.metrics package. In the plot below, the rows are the true labels, and the columns are predicted labels. Label 0 means no diabetic retinopathy and label 5 means proliferative diabetic retinopathy.

**Customized accuracy measure**: the accuracy score only tells us whether the predictions match with the true labels exactly. However, for a misclassified data point, we also want to know how far away is the predicted label from the true label. If the prediction is only 1 class away from the ground truth, this is better if it is 4 classes away from the ground truth. Therefore, we developed a customized accuracy measure (average distance in the plots below) to represent this logic.

**AUC, MCC**: we also looked into AUC and MCC, also using the respective functions from the sklearn package, to measure the effectiveness of the model to gain a better understanding of the misclassifications. The Matthews correlation coefficient (MCC) gives a number between -1 and 1 to summarize the confusion matrix, and the area under the curve graph (AUC) measures the model’s accuracy between 0 and 1, where 1 means the model fitted everything correctly and 0 means the model fitted entirely incorrectly.

## Results and Discussion

![12](https://github.gatech.edu/storage/user/54998/files/289dbd35-c90f-4548-95b7-8858b992b60f)<br/>
***Table 2. Table of comprehensive experiments with results.***
*Average distance is a customized metric we defined.*

### Effects of normalization
After correcting the pixel normalization logic, we saw a significant improvement in terms of model metrics. When we used the wrong normalization logic in the interim, the test accuracy of VGG16 and ResNet18 was 25% and 29%, respectively. However, the test accuracy of VGG16 and ResNet18 with the correct normalization logic is 38.6% and 40.8% respectively, which is around a 10% improvement. 
However, when we compare the correct normalized models with the non-normalized baseline models, we do not see a noticeable difference. The test accuracy of VGG16 was 33.2% without normalization, and 38.6% with normalization, where we observed a 5.4% improvement. The test accuracy of ResNet18 was 42.4% without normalization, and 40.8% with normalization, where we observed a 1.6% loss. When comparing the normalized model and non-normalized models, all other metrics remained on the same level as well. Therefore, we concluded that normalization does not help with model performance. This points to the fact that other image preprocessing techniques could be employed to enhance the performance.

### Effects of different learning rate
#### Lr = 0.01
Lr = 0.01 caused the loss to increase to infinity (NaN) within the first epoch. Adjusting Lr to 0.005 had the same result, albeit the increase was slightly slower than Lr = 0.01. This result is due to the high learning rate causing the model to create updates that are too large and lead to divergence in the loss.

Further decreasing the Lr to 0.0025 caused the loss to be stable, but the model did not learn very successfully. The final test accuracy was 0.2, and since the model is classifying images into five classes, this points to the model being only as good as randomly guessing. Looking at the results of the confusion matrices after each epoch, the images have a certain probability of getting classified into each of the classes that does not discriminate based on the image’s actual label, and this probability distribution changes between the epochs. We can see that though the loss does not become NaN, the model’s high learning rate is still causing it to fluctuate around the best solution but never arrive at it.

#### Lr = 0.001 
After changing the learning rate from 0.001 to 0.0001, the non-normalized VGG16 model has test accuracy increased from 33.2% to 34.8%. Although the amount increased is not significant, we can still see an increase in the model testing accuracy. We assume that it is due to the fact that a smaller learning rate may allow the model to learn a more optimal set of weights. Generally, a large learning rate allows the model to learn faster, but might make the model arrive on a sub-optimal final set of weights. Additionally, we did observe a longer training time when the learning rate was modified from 0.001 to 0.0001.

### Effects of different batch size
#### Batch size = 32
Decreasing the batch size of the non-normalized VGG-16 model decreased the test accuracy from 33.2% to 30.6%. Although a batch size of 32 has advantages such as requiring less memory and faster training time, the smaller batch outputs a less accurate estimate. From this experiment, we concluded that 32 may have been too small of a size. 

#### Batch size = 64
Increasing the batch size of the non-normalized VGG16 from 50 to 63 did not increase the test accuracy of the model either. In fact, the test accuracy of the model dropped from 33.2% to 32.8%. Although compared with the results of the non-normalized VGG16 model that has a batch size of 32, the test accuracy of the model with a batch size of 64 did not drop as much, there was no improvement on the model performance. We may conclude that increasing the batch size did not improve the model performance. Although a larger batch size may improve the effectiveness of the optimization steps resulting in more rapid convergence of the model parameters, we suspect that the larger batch size might make the model arrive on a sub-optimal final set of weights in this case.

### Conclusion
In summary, our team experimented with two different pre-trained PyTorch models (VGG16 and ResNet18) as an attempt to classify five severity levels of diabetic retinopathy based on retinal fundus images. Specifically, we experimented with the effects of pixel normalization, different learning rates, and different batch sizes on the performance of the models. Based on the model results, we concluded that neither did pixel normalization nor the changes in learning rate and batch size do not yield a significant improvement.  

As shown in the table above, the accuracy of all the models we experimented with remained around 30-40% and the predicted labels were 1 class away from the actual label on average. Since the accuracy is relatively low, the models we developed cannot be applied to detect diabetic retinopathy in a real-world setting. We believe that the low accuracy is mainly due to a lack of computing power we have: we were only able to select 2,800 out of 35,127 available training images to train our models with Google Colab. Thus, future studies could focus on training with the whole dataset if given sufficient resources. Moreover, we only experimented with two pre-trained PyTorch models, but there exist many other pre-trained machine learning models that are suitable for this image classification problem. Therefore, future studies could potentially explore other models which might yield better results. Lastly, the trained model was a classification model, which classified the images to either of five classes. However, we believe that the line that distinguishes the distribution of different classes might not be clear enough for the model to perform well, especially under limited computing resources. Therefore, future studies could examine how the accuracy and result changes when training the model as a linear regression model.
 

## Appendix
![7](https://github.gatech.edu/storage/user/54998/files/93d4d371-9fca-412e-aea2-34bf99d7ad58)
![8](https://github.gatech.edu/storage/user/54998/files/75c5e801-c0f7-4532-b8b3-86a75f6c9a61)
![9](https://github.gatech.edu/storage/user/54998/files/353c5438-28c5-4c40-937c-91714798148d)<br/>
***Figure 3. Model accuracy, average distance, and loss graph for Baseline VGG 16 trained with non-normalized dataset.***

![16](https://github.gatech.edu/storage/user/54998/files/77970be2-6d10-4232-8358-56db0429a7bf)
![17](https://github.gatech.edu/storage/user/54998/files/8cf173e9-5a83-4d7e-a96e-6d20bb282eda)
![18](https://github.gatech.edu/storage/user/54998/files/0a33a8db-f214-43b5-b537-3fe85f3d2ed1)<br/>
***Figure 4. Model accuracy, average distance, and loss graph for ResNet 18 trained with normalized dataset.***

![unnamed (3)](https://github.gatech.edu/storage/user/54998/files/e26d3725-d65f-42cb-9be2-eb1964861943)
![unnamed (4)](https://github.gatech.edu/storage/user/54998/files/3de8d382-dd3d-4241-a81b-76a733ebd927)
![unnamed (5)](https://github.gatech.edu/storage/user/54998/files/bb563f2e-5ea5-4e66-be47-e9d3de5cb9f9)<br/>
***Figure 5. Model accuracy, average distance, and loss graph for VGG 16 trained with correctly normalized dataset.***

![unnamed (6)](https://github.gatech.edu/storage/user/54998/files/398d2ac7-c18b-4e13-af26-198c91ee84fa)<br/>
***Figure 6. Confusion matrix of VGG 16 trained with correctly normalized dataset. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![unnamed (7)](https://github.gatech.edu/storage/user/54998/files/ef56d2b7-bfa2-4e42-a40c-d4c90714e863)
![unnamed (8)](https://github.gatech.edu/storage/user/54998/files/5e799676-ae24-4ace-be5d-f5113a020028)
![unnamed (9)](https://github.gatech.edu/storage/user/54998/files/99a11589-b122-4c91-b736-666382c232ab)<br/>
***Figure 7. Model accuracy, average distance, and loss graph for ResNet 18 trained with correctly normalized dataset.***

![unnamed (10)](https://github.gatech.edu/storage/user/54998/files/efc005d8-04c7-4ea1-959c-fcd920990339)<br/>
***Figure 8. Confusion matrix of ResNet18 trained with correctly normalized dataset. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![unnamed (19)](https://github.gatech.edu/storage/user/54998/files/e3984068-abb3-49a4-b573-febfeae8ca5a)
![unnamed (20)](https://github.gatech.edu/storage/user/54998/files/5196d932-9717-407e-952c-6108efa08dc2)
![unnamed (21)](https://github.gatech.edu/storage/user/54998/files/326c36c4-5a30-4e2c-be12-519bbd3e0181)
<br/>
***Figure 9. Model accuracy, average distance, and loss graph for VGG 16 trained with non-normalized dataset with Lr of 0.01.***

![unnamed (22)](https://github.gatech.edu/storage/user/54998/files/e4ed7b30-2db4-4d73-b4bf-a92af759d80e)
<br/>
***Figure 10. Confusion matrix of VGG 16 trained with non-normalized dataset with Lr of 0.01. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![unnamed (23)](https://github.gatech.edu/storage/user/54998/files/c62e9952-56b6-468b-bedc-5c2410b4c3c9)
![unnamed (24)](https://github.gatech.edu/storage/user/54998/files/d846860d-b29b-4642-b525-18bc3f02adae)
![unnamed (25)](https://github.gatech.edu/storage/user/54998/files/4a7bebd5-7241-42f9-b9c4-c67ad6cd7a6c)
<br/>
***Figure 11. Model accuracy, average distance, and loss graph for VGG 16 trained with non-normalized dataset with Lr of 0.0001.***

![unnamed (26)](https://github.gatech.edu/storage/user/54998/files/f9040043-c804-4ba1-830b-af120243143f)
<br/>
***Figure 12. Confusion matrix of VGG 16 trained with non-normalized dataset with Lr of 0.0001. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![unnamed (11)](https://github.gatech.edu/storage/user/54998/files/76570b88-1590-4e34-96f9-abc6b0ac1a0a)
![unnamed (12)](https://github.gatech.edu/storage/user/54998/files/6b02d1f0-7f04-49ef-9ed1-2d091d125387)
![unnamed (13)](https://github.gatech.edu/storage/user/54998/files/2972b6d9-7d3b-46fb-a37e-b72197ff440e)<br/>
***Figure 13. Model accuracy, average distance, and loss graph for VGG 16 trained with non-normalized dataset with batch size of 32.***

![unnamed (14)](https://github.gatech.edu/storage/user/54998/files/4c941104-04cb-4553-874d-9c43d8fe0af8)<br/>
***Figure 14. Confusion matrix of VGG 16 trained with non-normalized dataset with batch size of 32. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![unnamed (15)](https://github.gatech.edu/storage/user/54998/files/aab0387b-c1c6-4c9e-84db-732124cd99f4)
![unnamed (16)](https://github.gatech.edu/storage/user/54998/files/4e3b7527-6ae4-454e-abe5-baab9a5ca51e)
![unnamed (17)](https://github.gatech.edu/storage/user/54998/files/8d9064a5-ad13-4446-bef8-9e10b96514d7)
<br/>
***Figure 15. Model accuracy, average distance, and loss graph for VGG 16 trained with non-normalized dataset with batch size of 64.***

![unnamed (18)](https://github.gatech.edu/storage/user/54998/files/5e241ec5-72e4-4e5f-8961-c001224e65ec)
<br/>
***Figure 16. Confusion matrix of VGG 16 trained with non-normalized dataset with batch size of 64. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![19](https://github.gatech.edu/storage/user/54998/files/1b13139e-8c82-4d4c-bd2d-96a071e1c228)<br/>
***Figure 17. Confusion matrix of Baseline VGG 16 trained with non-normalized dataset. X-axis indicates the predicted labels. Y-axis indicates the true labels.***

![21](https://github.gatech.edu/storage/user/54998/files/b01d05be-7902-4589-81a7-1d040364b886)<br/>
***Figure 18. Confusion matrix of ResNet 18 trained with non-normalized dataset. X-axis indicates the predicted labels. Y-axis indicates the true labels.***


## References
U.S. Department of Health and Human Services. (n.d.). Diabetic retinopathy data and statistics. National Eye Institute. Retrieved October 2, 2021, from https://www.nei.nih.gov/learn-about-eye-health/outreach-campaigns-and-resources/eye-health-data-and-statistics/diabetic-retinopathy-data-and-statistics. 

Mayo Foundation for Medical Education and Research. (2021). Diabetic retinopathy. Mayo Clinic. Retrieved October 2, 2021, from https://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/symptoms-causes/syc-20371611. 

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

Gulshan, Varun, et al. "Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs." Jama 316.22 (2016): 2402-2410.
Simonyan, K., & Zisserman, A. (2014). Very dee
