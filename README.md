# IEEE Open Journal of the Communications Society
## Title: A Collaborative Software Defined Network-based Smart Grid Intrusion Detection System (ICC Extension)
### Abstract:
Current technological advancements including Software Defined Networks (SDN) can provide
better solutions in the topic of smart grids (SGs). An SDN-based SG promises to enhance the efficiency, reliability and sustainability of the communication network. However, new security breaches can be introduced with this adaptation. A layer of defence against insider attacks can be established using machine learning based intrusion detection system (IDS) located on the SDN application layer. Conventional centralised practises, violate the user data privacy aspect, thus distributed or collaborative approaches can be adapted so that attacks can be detected and actions can be taken. This paper proposes an new SDN-SG architecture highlighting the existence of IDSs in the SDN application layer. Secondly we implemented a new smart meter (SM) collaborative intrusion detection system (SM-IDS), using split learning methodology.  Finally, a comparison of a federated learning and split learning neighbourhood area network (NAN) IDS was made. Numerical results showed, a five class classification accuracy of over  80.3\%  and F1-score 78.9 for a SM-IDS adapting the split layer methodology. Also, the split learning NAN-IDS exhibit an accuracy of over 81.1\%  and F1-score 79.9.  

###  SDN-based SG Architecture:
![plot](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Figures/SDN_based_SG_Approach_new.png)

Data traffic is captured, and passed through the pre-processing stage, and then either split learning or federated learning is used.

  

## Helpful Documentation
### Installation 
It is highly recommended to use Colaboratory ([Colab](https://colab.research.google.com/notebooks/welcome.ipynb)) to run the notebooks, because it allows to write and execute Python code in a browser with:

- Zero configuration required
- Free access to GPUs and TPUs
- Most libraries pre-installed
- Only one requirement, a google account
- Most common Machine Learning frameworks pre-installed and ready to use

> Note: if you are not going to use Google Colab you will need to make sure that you satisfy the below requirements

#### Requirements
- SNNtorch (>= 0.5.1)
- PyTorch (>= 1.11.0)
- Numpy (>= 1.21.6)
- Pandas (>= 1.3.5)
- Seaborn (>= 0.11.2)
- Matplotlib (>= 3.2.2)
- Sklearn (>= 1.0.2)
- Flwr (== 0.19.0)
- Openml (== 0.12.2)

### Usage 
  #### Prepare Data
  In order to prepare your data follow the steps below:

  1. Download one of the following scripts depending on the desired experiment
[binary_classification_std_scaler](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/BinaryClassification/nsl_kdd_preprocessing_binary_stdscaler.ipynb),
[binary_classification_minmax_scaler](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/BinaryClassification/nsl_kdd_preprocessing_binary_minmaxscaler.ipynb),
[multiclass_classification_std_scaler](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/MultiClassClassification/nsl_kdd_preprocessing_multiclass_stdscaler.ipynb),
[multiclass_classification_minmax_scaler](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/MultiClassClassification/nsl_kdd_preprocessing_multiclass_minmaxscaler.ipynb)
  > Note: Alternatively launch the desired script using the launch button 

  2. If you want to process the NSLKDD dataset in a different way you can download it from [here](https://www.unb.ca/cic/datasets/nsl.html)
    
  3. Open [Colab](https://colab.research.google.com/notebooks/welcome.ipynb) and sign in to your Google account. If you do not have a Google account, you can create one [here](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp).

  4. Go to _File > Upload notebook > Choose file_ and browse to find the downloaded notebook. If you have already uploaded the notebook to Colab you can open it with _File > Open notebook_ and choose the desired notebook. 
  
#### Spiking Neural Network
In order to train a SNN model follow the steps below:

1. Download the [spiking_neural_network.ipynb](https://github.com/sotirischatzimiltis/MscThesis/blob/main/SpikingNeuralNetwork/spiking_neural_network.ipynb ).
> Note: Alternatively launch the **spiking_neural_network.ipynb** through the launch button
2. Open [Colab](https://colab.research.google.com/notebooks/welcome.ipynb) and sign in to your Google account. If you do not have a Google account, you can create one [here](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp).

3. Go to _File > Upload notebook > Choose file_ and browse to find the downloaded notebook file [spiking_neural_network.ipynb](https://github.com/sotirischatzimiltis/MscThesis/blob/main/SpikingNeuralNetwork/spiking_neural_network.ipynb ). If you have already uploaded the notebook to Colab you can open it with _File > Open notebook_ and choose **spkiking_neural_network.ipynb**. 

3. Once the notebook is loaded, go to _Runtime > Change runtime type_ and from the dropdown menu, under **Hardware accelerator**, choose **GPU** and click **Save**.

5. Now you can begin the experiments. All you have to do is to upload the dataset you want and set the parameters in the cell under **Datasets** section.

6. To train the model go to _Runtime > Run all_ or click on the first cell and use **Shift + Enter** to execute each cell one by one.

7. The hyper parameters of the model can be modified in the cell under **Set Train Arguments** section.

##### Set Train Arguments
1. bsize: Batch Size
2. nhidden: Number of hidden nodes
3. nsteps: Number of input time steps
4. b: beta/decay factor of membrane potential 
5. learning_rate: Learninig Rate of optimizer
6. nepochs: Number of training epochs
 
 #### Traditional ML techniques
 1. Download either the [binary](https://github.com/sotirischatzimiltis/MscThesis/blob/main/TraditionalML/traditionalml_binary_classification.py) or [multiclass](https://github.com/sotirischatzimiltis/MscThesis/blob/main/TraditionalML/traditionalml_multiclass_classification.py) classification python script.

 2. Put the correct paths to the test and train datasets.
 
 3. Execute the script.
  > Note: No need to assign values to hyperparameters. The script uses gridsearchCV using two-fold cross validation to find the best hyperparameters from a given list 
 
 #### Federated Learning
 > Note: For this part it is better to follow the documentation provided by Flower [here](https://flower.dev/docs/)
 1. Open the experiment folder you want to recreate: [MLP](https://github.com/sotirischatzimiltis/MscThesis/tree/main/FederatedLearning/MLP) , [SNN](https://github.com/sotirischatzimiltis/MscThesis/tree/main/FederatedLearning/SNN) or [LogReg](https://github.com/sotirischatzimiltis/MscThesis/tree/main/FederatedLearning/LogReg).  
 2. Download the files of the experiment.
    > Note: The client and server scripts do not contain RSA encryption. For utilising encryption as well follow the documentation provided by flower
 3. Open the terminal and make sure you satisfy the requirements needed to run the experiments
 4. Set server variables: 
     - Set global test path 
     - MLP: network variables & batch_size
     - SNN: network variables & batch_size
     - LogReg: alter logistic regression parameters if you wish
 5. Set client variables:
      - Set local train and test paths
      - MLP: network variables & batch_size
      - SNN: network variable & batch_size
      - LogReg: alter logistic regression parameters if you wish
 6. One terminal is needed for the server and one terminal is needed for every client (alternatively a script can be created) 
