# HAABSA + Quantification
This is the repository for replicating the fidngings from ABSQ 2021 paper. 

# HAABSA++
The code for A Hybrid Approach for Aspect-Based Sentiment Analysis Using Contextual Word Emmbeddings and Hierarchical Attention

The hybrid approach for aspect-based sentiment analysis (HAABSA) is a two-step method that classifies target sentiments using a domain sentiment ontology and a Multi-Hop LCR-Rot model as backup.
 - HAABSA paper: https://personal.eur.nl/frasincar/papers/ESWC2019/eswc2019.pdf
 
 Keeping the ontology, we optimise the embedding layer of the backup neural network with context-dependent word embeddings and integrate hierarchical attention in the model's architecture (HAABSA++).
 
 ## Software
The HAABSA source code: https://github.com/ofwallaart/HAABSA needs to be installed. Then the following changes need to be done:
- Update files: config.py, att_layer.py, main.py, main_cross.py and main_hyper.py.
- Add files: 
  - Context-dependent word embeddings: 
    - getBERTusingColab.py (extract the BERT word embeddings);
    - prepareBERT.py (prepare the final BERT emebdding matrix, training and tesing datasets);
    - prepareELMo.py (extract the ELMo word emebddings and prepare the final ELMo embedding matrix, training and testing datasets);
    - raw_data2015.txt, raw_data2016.txt (Data folder).
  - Hierarchical Attention: 
    - lcrModelAlt_hierarchical_v1 (first method);
    - lcrModelAlt_hierarchical_v2 (second method);
    - lcrModelAlt_hierarchical_v3 (third method);
    - lcrModelAlt_hierarchical_v4 (fourth method).

The training and testing datasets are in the Data folder for SemEval 2015 and SemEval 2016. The files are available for Glove, ELMo and BERT word emebddings. 

*Even if the model is trained with contextul word emebddings, the ontology has to run on a dataset special designed for the non-contextual case.


## AspEntQuaNet

This appendix will explain how to navigate in the code structure.

1. `main.py`

     This file represent the starting point of replicating all the experiments. Currently, all the default options are correct (everything is false but `runLCRROTALT_v4` is True).

2. `lcrModelAlt_hierarchical_v4.py`

     This is one of the first files called by main.py. It trains the LCR-Rot-hop++ for either the full dataset or a certain aspect category. Following that, it saves the necessary hidden states as well as probabilities in the `.txt` files. 

3. `config.py`

     This is the file which contains all the parameters for the trained LCR-Rot-hop++. The current default parameters are the ones that were used to train the models in the paper. 

4. `QuaNet.py`

     The next step is to start this file which is responsible to train QuaNet. It is important to remember to change the location paths of the training and testing datasets. By changing the values of the constants in the beginning of the file, it is possible to change the parameters of the models and the way they are trained. 

5. `utils.py`

     This is the file with all the useful functions that might be relevant to multiple classes. In particular, it includes the `calculate_classifications` method which is supposed to return the results of CC, ACC, PCC, and PACC. Additionally, it returns the ratio of false and true prediction rates. 

6. `test.py`

     This file is responsible for model evaluation on the data particular to certain aspect categories. The dataframe called `results\-losses` contains the results of all models (CC, ACC, PCC, PACC, QuaNet, EntQuaNet, AspEntQuaNet) for each of the evaluation measures. 

7. `graphs.py`

     This file contains the code that generates all the graphs present in this thesis. 



 ## Word embeddings
 - GloVe word embeddings (SemEval 2015): https://drive.google.com/file/d/14Gn-gkZDuTVSOFRPNqJeQABQxu-bZ5Tu/view?usp=sharing
 - GloVe word embeddings (SemEval 2016): https://drive.google.com/file/d/1UUUrlF_RuzQYIw_Jk_T40IyIs-fy7W92/view?usp=sharing
 - ELMo word embeddings (SemEval 2015): https://drive.google.com/file/d/1GfHKLmbiBEkATkeNmJq7CyXGo61aoY2l/view?usp=sharing
 - ELMo word embeddings (SemEval 2016): https://drive.google.com/file/d/1OT_1p55LNc4vxc0IZksSj2PmFraUIlRD/view?usp=sharing
 - BERT word embeddings (SemEval 2015): https://drive.google.com/file/d/1-P1LjDfwPhlt3UZhFIcdLQyEHFuorokx/view?usp=sharing
 - BERT word embeddings (SemEval 2016): https://drive.google.com/file/d/1eOc0pgbjGA-JVIx4jdA3m1xeYaf0xsx2/view?usp=sharing
 
Download pre-trained word emebddings: 
- GloVe: https://nlp.stanford.edu/projects/glove/
- Word2vec: https://code.google.com/archive/p/word2vec/
- FastText: https://fasttext.cc/docs/en/english-vectors.html
