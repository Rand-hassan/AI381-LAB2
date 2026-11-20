\# Model Selection and Promotion Policy



\## 1. Candidate Models



Three candidate models were trained and evaluated using the same dataset split:



\- \*\*Logistic Regression:\*\* A simple and lightweight baseline model.

\- \*\*Random Forest:\*\* An ensemble model capable of capturing nonlinear relationships.

\- \*\*Gradient Boosting:\*\* An ensemble model that combines sequential weak learners.



\## 2. Experimental Results



| Model | AUC | F1 | P95 Latency (ms) | Model Size (KB) |

|---|---:|---:|---:|---:|

| Logistic Regression | 0.7833 | 0.5124 | 7.49 | 3.55 |

| Random Forest | 0.7593 | 0.5289 | 27.72 | 3105.75 |

| Gradient Boosting | 0.7753 | 0.4960 | 7.63 | 140.69 |



\## 3. Selection Criteria



The promotion gate selects the candidate with the highest F1 score among models that satisfy the latency requirement.



\- \*\*Latency Budget:\*\* 25 ms.

\- \*\*Minimum F1 Improvement:\*\* 0.01.



The latency budget was selected to keep prediction latency low while allowing efficient models to compete. Random Forest exceeded this limit and was therefore excluded.



The minimum F1 improvement of 0.01 prevents replacing the current champion for a negligible improvement.



\## 4. Champion Selection



Logistic Regression was selected as the initial champion.



Although Random Forest achieved the highest overall F1 score, its P95 latency exceeded the allowed budget.



Among the eligible models, Logistic Regression achieved the highest F1 score (0.5124), slightly outperforming Gradient Boosting (0.4960). It also had the highest AUC and the smallest model size.



Therefore, Logistic Regression provides the best balance of predictive performance, inference speed, and model size for this project.



\## 5. Promotion Policy



A candidate can replace the current champion only if:



1\. Its P95 latency is within the 25 ms budget.

2\. Its F1 score is at least 0.01 higher than the current champion's F1.



The selected model is registered in MLflow and assigned the `champion` alias. Its model artifact and metadata are exported to the `artifacts/` directory.

