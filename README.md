
Data_preprocessing_and_feature_selction.py contains Python code for preprocessing data and performing feature selection for regression model

Clustering_Feature_Generation.py contains Python code for Unsupervised representaion of the MMG dataset to obtain cluster information and then further in-depth analysis of MMG compositions based on unsupervised clustering to be used as additional features for symbolic regression (DESyR) model.

Regression_GFA_Metallic_Glasses.py contains Python code for performing regression analysis on glass-forming ability (GFA) of multicomponent metallic glasses. Multiple regression models, including Random Forest, Gradient Boosting, SVM, Lasso, Ridge, ElasticNet, K-Nearest Neighbors, and neural networks, are applied to predict the target values (Tg, Tx, and Tl).

Standard_Symbolic_Regression.py contains Python code for standard symbolic regression using genetic programming to model the glass-forming ability (GFA) or related properties. Additionally, model complexity is analyzed and optimized in competition with model accuracy.

Dimensions_embedded_Symbolic_Regression.py contains Python code for dimensions embedded symbolic regression using genetic programming to model the glass-forming ability (GFA) or related properties. Additionally, model complexity is analyzed and optimized in competition with model accuracy.

Generate_new_compostions_GAN.py contains Python code for generating new MMG compositions using a Generative Adversarial Network (GAN). The code defines methods for constructing the generator and discriminator hybrid model for the GAN framework. These models work to generate and refine new compositions based on latent space representations.

Classifier_check_for_generated_compositions.py contains Python code for building and evaluating classification models for multicomponent alloy compositions. Several classifiers are defined, including Random Forest, Gradient Boosting, SVM, Logistic Regression, K-Nearest Neighbors, Naive Bayes, and a Neural Network model built built to classify multicomponent metallic glasses from crystalline alloys.
