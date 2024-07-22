# IEEE Open Journal of the Communications Society
## Title: A Collaborative Software Defined Network-based Smart Grid Intrusion Detection System (ICC Extension)
### Abstract:
Current technological advancements including Software Defined Networks (SDN) can provide
better solutions in the topic of smart grids (SGs). An SDN-based SG promises to enhance the efficiency, reliability and sustainability of the communication network. However, new security breaches can be introduced with this adaptation. A layer of defence against insider attacks can be established using machine learning based intrusion detection system (IDS) located on the SDN application layer. Conventional centralised practises, violate the user data privacy aspect, thus distributed or collaborative approaches can be adapted so that attacks can be detected and actions can be taken. This paper proposes an new SDN-SG architecture highlighting the existence of IDSs in the SDN application layer. Secondly we implemented a new smart meter (SM) collaborative intrusion detection system (SM-IDS), using split learning methodology.  Finally, a comparison of a federated learning and split learning neighbourhood area network (NAN) IDS was made. Numerical results showed, a five class classification accuracy of over  80.3\%  and F1-score 78.9 for a SM-IDS adapting the split layer methodology. Also, the split learning NAN-IDS exhibit an accuracy of over 81.1\%  and F1-score 79.9.  

###  SDN-based SG Architecture:
![plot](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Figures/SDN_based_SG_Approach_new.png)

Data traffic is captured, and passed through the pre-processing stage, and then either split learning or federated learning is used.

### Usage 
  #### Prepare Data
  In order to prepare your data follow the steps below:

  1. Download the following script [nsl_kdd_multiclass_preprocessing.py](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/nsl_kdd_multiclass_preprocessing.py),

  2. If you want to process the NSLKDD dataset in a different way you can download it from [here](https://www.unb.ca/cic/datasets/nsl.html)
  

 
 #### Split Learning
 1. Split learning can be applied with one or more clients participating.
 2. In case you want to use one client and a server we can download the following scripts: [label sharing](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/split_learning_one_client_script.py), [no label sharing](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/split_learning_no_label_sharing_one_client.py)
 3. For multiple clients download the specific scripts: [label sharing](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/traditional_split_learning_withvalidation.py), [no label sharing](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/traditional_split_learning_no_label_sharing_withvalidation.py)
> Note: [model.py](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/models.py)  auxiliary script should be downloaded 
 
 #### Federated Learning
 > Note: For this part it is better to follow the documentation provided by Flower [here](https://flower.dev/docs/)
 1. Download the file necessary: [client](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/fl_client_script.py), [server](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/fl_server_script.py)
> Note: [model.py](https://github.com/sotirischatzimiltis/IEEE_OJ_COMS_SG_IDS/blob/main/Scripts/models.py)  auxiliary script should be downloaded 
