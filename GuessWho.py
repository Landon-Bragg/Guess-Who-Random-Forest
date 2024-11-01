import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Data/Guess Who Data.csv")



# Function to calculate the average depth of the leaves
def calculate_average_leaf_depth(tree):
    node_depth = np.zeros(shape=tree.tree_.node_count)
    stack = [(0, 0)]  # (node_id, depth)
    while stack:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        left_child = tree.tree_.children_left[node_id]
        right_child = tree.tree_.children_right[node_id]
        if left_child != right_child:  # Not a leaf node
            stack.append((left_child, depth + 1))
            stack.append((right_child, depth + 1))

    leaf_depths = node_depth[tree.tree_.children_left == tree.tree_.children_right]
    average_leaf_depth = np.mean(leaf_depths)
    return average_leaf_depth

# Function to evaluate the average leaf depth for a given random state
def evaluate_random_state(random_state, X, y):
    dt = DecisionTreeClassifier(random_state=random_state, max_depth=None)
    dt.fit(X, y)
    avg_leaf_depth = calculate_average_leaf_depth(dt)
    return avg_leaf_depth

# Separate features and target
X = data.drop(['Name'], axis=1)
y = data['Name']

# Iterate over different random states to find the optimal one
random_states = range(0, 100000)  # Define the range of random states to test
results = []

for state in random_states:
    avg_leaf_depth = evaluate_random_state(state, X, y)
    results.append((state, avg_leaf_depth))
    print(f"Random state {state} - Average leaf depth: {avg_leaf_depth:.2f}")

# Convert results to a DataFrame for better readability
results_df = pd.DataFrame(results, columns=['Random State', 'Average Leaf Depth'])

# Find the optimal random state
optimal_state = results_df.loc[results_df['Average Leaf Depth'].idxmin()]
print(f"\nOptimal random state: {optimal_state['Random State']} with average leaf depth: {optimal_state['Average Leaf Depth']:.2f}")

# Optionally, visualize the decision tree for the optimal random state
optimal_dt = DecisionTreeClassifier(random_state=optimal_state['Random State'], max_depth=None)
optimal_dt.fit(X, y)






# Separate features and target
X = data.drop(['Name'], axis=1)
y = data['Name']

# Create and fit the decision tree
dt = DecisionTreeClassifier(random_state=166467, max_depth=None)
dt.fit(X, y)

# Get the depth of the tree for optimization purposes
depth = dt.get_depth()

# Make the plot bigger to be able to read names
plt.figure(figsize=(80, 50))

# Plot the decision tree
class_names = list(y.unique())
plot_tree(dt, feature_names=X.columns.tolist(), class_names=class_names, filled=True, rounded=True, fontsize=8)

# Print the depth of the tree
print(f"Depth of the decision tree: {depth}")

# Print the feature importances
importances = pd.DataFrame({'feature': X.columns, 'importance': dt.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
print("\nFeature importances:")
print(importances)

# Print information about the tree to ensure minimal nodes
n_nodes = dt.tree_.node_count
n_leaves = dt.tree_.n_leaves
print(f"\nTotal number of nodes: {n_nodes}")
print(f"Number of leaf nodes (terminal nodes): {n_leaves}")

# Calculate the average depth of the leaves
def calculate_average_leaf_depth(tree):
    # Get the depth of each leaf node
    node_depth = np.zeros(shape=tree.tree_.node_count)
    stack = [(0, 0)]  # (node_id, depth)
    while stack:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        left_child = tree.tree_.children_left[node_id]
        right_child = tree.tree_.children_right[node_id]
        if left_child != right_child:  # Not a leaf node
            stack.append((left_child, depth + 1))
            stack.append((right_child, depth + 1))

    # Average depth of the leaves
    leaf_depths = node_depth[tree.tree_.children_left == tree.tree_.children_right]
    average_leaf_depth = np.mean(leaf_depths)
    return average_leaf_depth

avg_leaf_depth = calculate_average_leaf_depth(dt)
print(f"\nAverage depth of the leaves: {avg_leaf_depth:.2f}")

# Make sure that all characters are reachable
leaf_labels = dt.apply(X)
unique_leaf_labels = np.unique(leaf_labels)
if len(unique_leaf_labels) == len(y):
    print("\nAll characters are uniquely reachable in the decision tree.")
else:
    print("\nWarning: Not all characters have a unique path in the decision tree.")
    unreachable = set(range(len(y))) - set(unique_leaf_labels)
    print(f"Characters not uniquely reachable: {[y.iloc[i] for i in unreachable]}")