# **Sub-document Neural Topic Model**

<!-- [Paper link]() -->

## **🚀 Getting Started**
Follow these steps to set up the project and run the models.

### **1. Prequisites**
Ensure you have the following installed:
1. Install `Python 3.11`

2. Install Java
    
    * [Download Java](https://www.java.com/en/download/)

### **2. Installation**
1. Install the required libraries
    ```bash
    pip install -r requirements.txt
    ```
2. Download [`tm_datasets`]()

3. Download `palmetto.jar`
    
    * Download [`palmetto.jar`]() and place it in the `evaluations/` directory.

4. Download Processed Wikipedia Corpus
    
    * Download and extract this [processed Wikipedia corpus]() and place it in the `evaluations/` directory as an external reference corpus

### **3. Directory Structure for `evaluations/`**
After setup, your evaluations/ folder should look like this:

```bash
├── evaluations
│   ├── llm_evaluation
│   │   ├── __init__.py
│   │   └── llm_eval.py
│   │   └── llm_model.py
│   ├── __init__.py
│   ├── topic_classification.py
│   ├── topic_clustering.py
│   ├── topic_coherence.py
│   ├── topic_diversity.py
│   ├── topic_inverted_bias_overlap.py
│   ├── palmetto.jar
│   ├── wikipedia_bd/
│   ├── wikipedia_bd.histogram
```
### **API Key Configuration**
Create a `.env` file in the root directory of the project and add your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
``` 

## **🚀 Usage**
### **1. Prepare Sub-documents**

Before running the model, you need to prepare sub-documents for each dataset.

Modify the `scripts/subdoc_prepare.sh` file with your desired dataset and parameters.

Example for `20NG` dataset:
```bash
python utils/subdoc_prepare.py \
--dataset_name 20NG \   
--window_size 40 \      
--stride 30 \
--batch_size 32 \
```
To execute the preparation script:
```bash
bash scripts/subdoc_prepare.sh
```

### **2. Run and Evaluate the Model**
After preparing the sub-documents, you can run and evaluate the model using the `scripts/runner.sh` script:
```bash
bash scripts/runner.sh
```


## **🚀 Acknowledgement**
This implementation builds upon the excellent work of:

* [TopMost](https://github.com/BobXWu/TopMost): Some parts of this implementation are based on TopMost
* [Palmetto](https://github.com/dice-group/Palmetto): Utilized for the evaluation of topic coherence.


<!-- ## **Citation**
```
@misc{}
``` -->

