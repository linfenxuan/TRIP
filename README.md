# TRIP
This is code of Type-Aware Information Injection for Inductive Link Prediction in Knowledge Graphs.
The primary objective of reasoning on Knowledge Graphs (KGs) is to derive novel facts based on existing ones. 
Inductive reasoning models have predominantly focused on predicting missing facts by acquiring logical rules. 
However, despite the significance of inductive relation prediction, most recent studies have been limited to a transductive framework, thereby lacking the capability to handle previously unseen entities. 
Nonetheless, an observation reveals that the subgraph mining methods often overlook the importance of entity types or the relational path, thereby limiting their comprehensive reasoning capabilities.

To address these challenges, we propose a novel approach called TRIP, which incorporates Type information injection and Relational Path learnIng for relation Prediction. 
In TRIP, we aim to enhance the modeling of subgraph representations in a comprehensive manner by leveraging mutual information for knowledge graphs. 
Through extensive experiments conducted on five versions of two fully-inductive datasets, TRIP outperforms all baseline methods in terms of predictive accuracy. 
Additionally, it achieves results that are comparable to state-of-the-art approaches in the remaining dataset versions. 
These experimental results validate the effectiveness of TRIP in exploring node neighboring relations on a global scale to characterize node features and reason through relational paths.

## Requiremnts
dgl==0.4.2
lmdb==0.98
networkx==2.4
scikit-learn==0.22.1
torch==1.4.0
tqdm==4.43.0

## Usage
Train data and test data are located in data folder.

### Training
python train.py -d WN18RR_v4 -e trip_wn_v4

### Evaluation
python test_auc.py -d WN18RR_v4_ind -e trip_wn_v4
python test_ranking.py -d WN18RR_v4_ind -e trip_wn_v4

### Acknowlegement
We refer to the code of GraIL. Thanks for their contributions.
