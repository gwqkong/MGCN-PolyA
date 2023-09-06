MGCN-PolyA
=======
## MGCN-PolyA: An integrated computational framework for predicting Poly(A) signals with multiscale gated convolutional networks  

  Polyadenylation is a crucial post-transcriptional modification that plays a critical role in protecting RNA stability and is involved in mRNA maturation and transcriptional regulation. r we design the prediction model named MGCN-PolyA, which is a predictor integrated by deep neural network and random forest. MGCN-PolyA extracts a variety of features, which are further extracted using two parallel convolutional neural networks, and multi-scale gated convolution is used in the convolutional neural network to improve the model's ability to understand and represent the input data. Finally, the output of the parallel convolutional neural network in full connection is used as the input of the random forest to obtain the final prediction results. In conclusion, MGCN-PolyA is a reliable predictive model in the PAS recognition problem. 
 
Requirements
=====
  Python=3.6, recommend installing Anaconda3  
  TensorFlow=2.6.2  
Keras=2.6.0  
numpy=1.19.2  
Biopython=1.79  
H5py, Hyperopt, Sklearn  

### -----------------------train model----------------------------------

The PAS training you can find in the 'data' folder.If you want to reuse the MGCN-polyA model, you first need to generate the required features using the code in the ***'features'*** folder, then run ***'MGCN1_w2v.py'*** and ***'MGCN2_oe.py'*** and finally run ***'rf_prediction.py'*** .  



### -----------------------Predciton----------------------------------

The PAS test data you can find in the **'data'** folder, and if you want to re-run the predictions using the model, you can run **'model test.py'** in the **'test'** folder .

If you have any queries, please email to gwq0846@163.com
