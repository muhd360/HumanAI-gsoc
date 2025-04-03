# HumanAI-gsoc
Colab link-https://colab.research.google.com/drive/1gI1vgmSAYQMcHuCaJxuWWsSj5g1P7Rtd?usp=sharing
Metrics for NER Model:-
(100 sample data points(manual annotation))
Precision-0.99
Recall-0.94
F1 Score-0.94 
For LDA Model(on sample set of 1000 from given data)
![Screenshot 2025-04-03 131040](https://github.com/user-attachments/assets/c37caab6-28a4-4aeb-8fca-65b1c8796451)

Only for label-5(MIC)
![Screenshot 2025-04-03 125505](https://github.com/user-attachments/assets/298131c2-5101-4eef-8056-937d33a3a3f6)

Calculated using 2 ways:-
Prompting claude and chatgpt api for review of article.
Manually reviewing article is MIC or not(More than 1k manual review was not feasible at my end).

Issues:-
High false positives in LDA
High accuracy of model attribute of less testing data.Counting days also as Date in some instances.Performs well in casualty(both numeric and text)
![image](https://github.com/user-attachments/assets/c1d8f2e7-130f-415c-b38d-5006f025ab95)

Further improvements:-
LDA Model corpus extended from 50k till entire 3 million(Hardware required)
Auto Annotation using LLM'S to finetune gliner model
