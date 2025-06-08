# GroupProject

# Traffic Sign Dataset: https://www.kaggle.com/datasets/daniildeltsov/traffic-signs-gtsrb-plus-162-custom-classes/data

# Submission of All Submissions

Milestone 2:

https://github.com/jaron-ucsd/GroupProject/blob/Milestone2/README.md

Milestone 3:

https://github.com/jaron-ucsd/GroupProject/blob/Milestone3/README.md

# Introduction

We chose to work with a traffic sign dataset because traffic sign recognition is a critical component of modern driver assistance systems and autonomous vehicles. It's a fascinating challenge that combines computer vision and real-world problem solving. What makes this project especially cool is its focus on imperfect, real-world conditions, such as blurry, tilted, faded, or partially blocked signs, rather than just clean, textbook examples. These kinds of edge cases are exactly where current systems can struggle the most.	

By training and evaluating models on these more realistic scenarios, we can better understand how model performance is affected by visual noise and distortion. The broader impact of developing a robust predictive model in this context is significant: improving traffic sign recognition can enhance road safety, reduce accidents, and support the safe deployment of autonomous vehicles. A model that performs well under difficult conditions isn't just a technical achievement, but it's also a step toward making AI more trustworthy and practical in everyday life.

# Written Report Link
https://docs.google.com/document/d/1a97W1LC6_tbtI3Hov0IUUtxh2RBH3cem5W3VpJwmbKw/edit?usp=sharing

# Final Summary and Results

Overall, the model demonstrated strong learning and generalization capabilities, with steady improvements across all training epochs and a final validation accuracy of 87.14%. Its performance was consistent across most of the 205 traffic sign classes, achieving a weighted F1-score of 0.87 and macro F1-score of 0.84 on the validation set. When evaluated on the held-out test set of over 53,000 samples, the model achieved a solid accuracy of 77.93%, confirming its ability to generalize to unseen data. While the majority of classes were classified accurately, many with F1-scores above 0.85, a small subset of classes remained difficult, likely due to data imbalance or visual similarity with other signs. These results suggest that the model is effective for large-scale traffic sign recognition, with room for improvement in handling underrepresented or ambiguous classes.

