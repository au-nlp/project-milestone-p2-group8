# Feedback for Project milestone P2

## Overall Feedback
Great milestone in general. However, there are some concerns that are detailed below. My main comments would be:  
1) Some details regarding the proposed fusion architecture are missing.  
2) Do you think the dataset includes all the factors needed for you to build a success prediction model? For example, financial or regulatory confounders are not there. Is it possible that the model can only predict trial completion based on partial information and may capture correlations rather than true causal drivers?

## Abstract
Clear motivation and good framing. Novelty might be a bit overstated, you need to emphasize why your approach is new. “Success” needs a more precise definition so the reader has a clear understanding of your pitch.

## Novelty
Good context and clear contributions. Novelty claims are a bit vague though without architectural details. Strengthen justification for why your fusion approach is unique, and clarify how insights will be extracted.

## Methods
**EDA:** Strong understanding of data quirks and imbalance.  
**Data Preparation and Pipeline:** Pipeline is sound. I don't get the “minimal missingness” justification. Also, I feel that here you should mention something about the target imbalance that you previously mentioned. How did you handle that?  
**Baseline Modeling:** Nice baselines. My only worry here is that an F1-score of 0.89 is almost suspiciously high, which might indicate label leakage or overly easy prediction. Of course, class imbalance plays a role in that. I think this issue must be fixed so that your conclusions are more meaningful.  
**Downsampling Experiment:** Strong experimental intuition → directly tests imbalance as a root cause.

## Timeline and organization within the team
The proposed timeline makes sense and is clear to me. I think 7–8 days should be more than enough to write the final report, but that's completely up to you.

Cool titles in the team organization section. My only comment here is that you should do this for Project Milestone P3 as well in advance. This allows you to plan better and divide responsibilities in a practical manner.

## Appendix
Detailed and nicely written. Great initiative to include the GenAI declaration!

## Question for TAs:
No questions

**Textual description quality:** Excellent work, keep it up!  
**Code quality:** Excellent work, keep it up!