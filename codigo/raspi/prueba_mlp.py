from mlp.mlp import DynamicMLP
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

# # Load the dataset
# filename = "electricity.csv"
# dataset = pd.read_csv("dataset/" + filename)
# dataset.replace({'UP': 1, 'DOWN': 0, 'True': 1, 'False': 0}, inplace=True)

# print(dataset.iloc[:20000, -1].value_counts())


# # Extract features and labels
# features = dataset.iloc[:, :-1]  # All columns except the last (target)
# labels = dataset.iloc[:, -1]  # The last column (target)


# # Hyperparameters
# feature_columns = dataset.columns[:-1]  # All except the last column


# # Initialize the MLP model
# model1 = DynamicMLP(
#     input_size=len(feature_columns),
#     hidden_units=hidden_units,
#     learning_rate=learning_rate,
#     dropout_rate=dropout_rate,
#     weight_decay=weight_decay
# )

# model2 = DynamicMLP(
#     input_size=len(feature_columns),
#     hidden_units=hidden_units,
#     learning_rate=learning_rate,
#     dropout_rate=dropout_rate,
#     weight_decay=weight_decay
# )

# model3 = DynamicMLP(
#     input_size=len(feature_columns),
#     hidden_units=hidden_units,
#     learning_rate=learning_rate,
#     dropout_rate=dropout_rate,
#     weight_decay=weight_decay
# )


# # Function to evaluate models and print confusion matrix
# def evaluate_model(model, features, labels, num_samples):
#     predictions = []
#     true_labels = []
#     for i in range(num_samples):
#         sample_idx = i % len(features)  # Loop over the dataset
#         sample = features.iloc[sample_idx].to_dict()
#         label = labels.iloc[sample_idx]
#         prediction = model.predict_one(sample)
#         predictions.append(prediction)
#         true_labels.append(label)
#     cm = confusion_matrix(true_labels, predictions)
#     accuracy = accuracy_score(true_labels, predictions)
#     precision = precision_score(true_labels, predictions)
#     f1 = f1_score(true_labels, predictions)
#     print(f"Confusion Matrix:\n{cm}")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     return cm, accuracy


# # Train and evaluate each model on 5000 samples (interleaved)
# # for i in range(5000):

# for i in range(15):
#     sample = features.iloc[i].to_dict()
#     label = labels.iloc[i]

#     # Predict-then-train for interleaved samples
#     if i % 3 == 0:
#         model1.predict_one(sample)
#         model1.learn_one(sample, label)
#     elif i % 3 == 1:
#         model2.predict_one(sample)
#         model2.learn_one(sample, label)
#     else:
#         model3.predict_one(sample)
#         model3.learn_one(sample, label)

# # Evaluate each model after training on 5000 samples
# print("Model 1 Evaluation:")
# cm1, acc1 = evaluate_model(model1, features, labels, 5000)

# print("\nModel 2 Evaluation:")
# cm2, acc2 = evaluate_model(model2, features, labels, 5000)

# print("\nModel 3 Evaluation:")
# cm3, acc3 = evaluate_model(model3, features, labels, 5000)

# # Aggregate parameters from model2 and model3 into model1
# params2 = model2.get_parameters()
# params3 = model3.get_parameters()
# model1.aggregate_and_update_parameters([params2])

# # Evaluate model1 after aggregation
# print("\nModel 1 Evaluation After Aggregation:")
# cm1_agg, acc1_agg = evaluate_model(model1, features, labels, 5000)

# # Train all models again on 5000 more samples (interleaved)
# for i in range(5000, 10000):
#     sample_idx = i % len(features)  # Loop over the dataset
#     sample = features.iloc[sample_idx].to_dict()
#     label = labels.iloc[sample_idx]

#     # Predict-then-train for interleaved samples
#     if i % 3 == 0:
#         model1.predict_one(sample)
#         model1.learn_one(sample, label)
#     elif i % 3 == 1:
#         model2.predict_one(sample)
#         model2.learn_one(sample, label)
#     else:
#         model3.predict_one(sample)
#         model3.learn_one(sample, label)

# # Evaluate each model after training on another 5000 samples
# print("\nModel 1 Evaluation After Additional Training:")
# cm1_final, acc1_final = evaluate_model(model1, features, labels, 5000)

# print("\nModel 2 Evaluation After Additional Training:")
# cm2_final, acc2_final = evaluate_model(model2, features, labels, 5000)

# print("\nModel 3 Evaluation After Additional Training:")
# cm3_final, acc3_final = evaluate_model(model3, features, labels, 5000)

# # Print final comparison
# print("\nFinal Comparison:")
# print(f"Model 1 Accuracy (After Aggregation): {acc1_agg:.4f}")
# print(f"Model 1 Accuracy (After Additional Training): {acc1_final:.4f}")
# print(f"Model 2 Accuracy (After Additional Training): {acc2_final:.4f}")
# print(f"Model 3 Accuracy (After Additional Training): {acc3_final:.4f}")



# Example usage
model = DynamicMLP(input_size=4, hidden_units=[32, 16], output_size=1)

# Sample input and label
sample = {0: 0.5, 1: 0.3, 2: 0.2, 3: 0.4}
label = 1

# Predict and learn
prediction = model.predict_one(sample)
loss = model.learn_one(sample, label)