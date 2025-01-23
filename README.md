# SESAME: Leveraging Labeled Overall Polarities for Aspect-based Sentiment Analysis based on SHAP-Enhanced Syntactic Analysis

<div align=center>
  <img src="https://github.com/nexpaliiz554/WWW2025/blob/main/schematicDiagram.png" width="500" >
</div>

In this paper, we propose an Aspect-Sentiment-Pair-Extraction (ASPE) approach that extracts aspect-sentiment pairs (a,s) through syntactic analysis guided by AI explainable framework, relying solely on the labeled overall sentiment polarities instead of additionally introduced manual annotations.
Specifically, our approach first leverages the overall-polarity-labeled data to train a sentiment classifier, and then deduces related aspects based on our syntactic analysis guided by the calculated contribution values for each word in the texts from our trained classifier. We named it as **SESAME** (**S**HAP-Guid**E**d **S**ytactic Analysis for **A**spect-based senti**ME**nt analysis). In particular, our approach consists of three stages. **First**, we use a RoBERTa-incorporated TextCNN framework to train a sentiment-polarity (i.e., positive, neutral, and negative) classifier. **Second**, we utilize the AI-Explanation framework SHAP (SHapley-Additive-exPlanation) to analyze the word's contribution (quantified as SHAP value) towards the predicted classification, and select representative words accordingly.  **Third**, we propose a set of syntactic patterns that consider word dependencies within each human-written sentence in the input texts, to extract aspects based on the representative words and accumulated frequent candidates within our analysis. The main contribution of our proposed SHAP-guided syntactic analysis lies in the following argument that *the exploited word dependencies can greatly reveal the implicit relations between the sentiment polarities and aspects, thus largely alleviating the manual efforts of additional human-annotated, intermediate sentiment elements, such as opinion and category.* As a result, the proposed approach is easy-to-use in practice for different domains compared to traditional approaches.

 Our evaluation is based on two English datasets  [[Cai et al.@ACL2021]](https://github.com/NUSTM/ACOS)  with the complete (𝑎,𝑐,𝑜,𝑠) quadruples annotated (for SESAME the quadruples are only used for evaluation), and two Chinese datasets [[Peng et al. @KBS2018]](http://sentic.net/chinese-review-datasets.zip) with the (𝑎,𝑠) pair annotated to show that SESAME can also support different languages when slightly adapted. Our four baseline approaches are three state-of-the-art (SOTA) learning-based ABSA approaches that consider (𝑎,𝑐,𝑜,𝑠) quadruples [Extract-Classify-ACOS](https://github.com/NUSTM/ACOS), (𝑎,𝑜,𝑠) triplets [Span-ASTE](https://github.com/chiayewken/Span-ASTE), and (𝑎,𝑠) pairs [LCF-ATEPC](https://github.com/yangheng95/LCF-ATEPC) (we used its implementation in [PyABSA](https://github.com/yangheng95/PyABSA/tree/release/demos/aspect_term_extraction) 2.3.3 during our experiments) respectively, and the recently popular ChatGPT (the underlying model was gpt-3.5-turbo-0613 for GPT 3.5 and gpt-4o-2024-08-06 for GPT 4o). For the ASPE task, our approach, which solely learns from overall-polarity-labeled data, can achieve on average 93.5% of the quadruple-learning approach in F1-score of precision and recall, 107.8% of the triple-learning approach, 119.4% of the pair-learning approach, 124.0% of ChatGPT 3.5, and 103.0% of ChatGPT 4o. The evaluation result shows that our approach is an easy-to-use and also explainable ABSA approach that achieves a comparable level of performance to SOTA learning-based approaches, and comprehensively outperforms ChatGPT by relying on coarse sentiment polarities only (i.e., one out of four manual labels required by the best-performing learning-based baseline), indicating that our approach is more applicable in specific domains (e.g., SE) where most datasets are only labeled by the overall sentiments.

 Since the benchmark dataset is not self-generated and published, we do not provide them here. If you want to use any of them, you should complete the license for their publication and consider referencing the original paper. You can download them from the link we provided above. (We publish the processed Chinese dataset in ```/Chinese AS Dataset``` folder of the project.)

## Overview

1. ```config.py``` The configurations of project.

2. ```EASTER_en.py```  Used for sentiment analysis of English text.

3. ```EASTER_ch.py``` Used for sentiment analysis of Chinese text.

   EASTER_en.py and EASTER_ch.py differ in :

   (1) They use different pretrained models and corresponding tokenizers.

   (2) In EASTER_ch.py, a CustomClassificationHead constructed to mimic the TFRobertaClassificationHead is used as the residual connections.

4. ```extract_opinion_en.py```  Calculate SHAP value, extract representative words for English text.

5. ```extract_opinion_ch.py```  Calculate SHAP value, extract representative words for Chinese text.

   extract_opinion_en.py and extract_opinion_ch.py differ in :

   (1) When computing SHAP values, they rely on the `load_model()` and `DataGenerator` specific to the corresponding language.

   (2) The parsing tags for different languages vary. For details, please refer to the [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) official website.

6. ```extrac_aspect.py```  Extracte aspects using syntactic rules.

7. ```data/sentiment_acos```  Files required for training and testing the sentiment classifier.

8. ```data/pretrained``` Store trained models

9. ```data/pred_senti```  Store the results of predicted sentiment

10. ```data/pred_opinion```  Store the output results of representative words 

11. ```data/pred_aspect```  Store the final extracted (a, s)

12. ```data/dict```  Store dictionary resources

13. ```data/acos```  Store manually annotated test set results

14. ```requirements.txt```  List the dependencies for the project

15. ```results.xlsx```  List the evaluation results in the paper


## Run

**Preparation**:

- *For the classifier in stage one*:

1. If you wish to reproduce our paper's data, you can download our pre-trained models from [NexpaLiiz554/SESAME on Hugging Face](https://huggingface.co/NexpaLiiz554/SESAME-WWW2025/tree/main) (the four models in the provided link serve as the foundational models for our ablation experiments), and place them in the ```data/pretrained```.
2. If you intend to retrain the classifier, for English, please download the pre-trained RoBERTa model from  [HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment/tree/main), and for Chinese, please download it from [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm).

- *For the extraction of representative words in the second stage*:

1. Download stanfordcorenlp tool from [CoreNLP](https://nlp.stanford.edu/software/stanford-corenlp-4.5.6.zip) (we used V4.5.6) 

2. For Chinese, you will need to download an additional Chinese model, copy both the Chinese model and StanfordCoreNLP-chinese.properties into STANFORD_CORE_NLP_PATH, and then run the following command under STANFORD_CORE_NLP_PATH to start the StanfordCoreNLP service:

   ```
   java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000
   ```

3. *Addition*: If you need to use ChatGPT’s NLP analysis as a supplement:  1) Use `POS_Tagging/gpt_pos_tagger.py` to obtain ChatGPT’s word segmentation and POS tagging results. 2) Use `POS_Tagging/data_process.py` to compare the outputs of ChatGPT and CoreNLP, and replace the analysis results of texts where ChatGPT has made clear errors (the output file will only include texts that need replacement and their corresponding ChatGPT NLP analysis results). 3) Place the generated file '{target}_tagged.txt' in the `data_WWW2025/gpt_POS_supplement` folder for further analysis.

  - *For aspect extraction in the third stage*:

    You can directly download SentiAspectExtractor.jar from  [nexpaliiz554/SESAME on Hugging Face](https://huggingface.co/NexpaLiiz554/SESAME-WWW2025/tree/main) . **Additionally**, we have open-sourced this code on GitHub at [nexpaliiz554/SentiAspectExtractor.](https://github.com/nexpaliiz554/SentiAspectExtractor)

    Finally, configure your config.py



**Running**:

1. run EASTER_en.py / EASTER_ch.py to to train or predict for your data.

2. run extract_opinion_en.py / extract_opinion_ch.py to select representative words

3. run extrac_aspect.py to extract aspects

4. run evaluate.py to assess result