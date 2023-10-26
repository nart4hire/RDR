from __future__ import annotations
import pandas as pd
import numpy as np
from random import choice, seed as set_seed


class RDR:
    def __init__(self, general_case: str | None = None) -> None:
        self.general_case: str | None = general_case
        self._root: RDR.RDRNode = self.RDRNode(None, None)

    def fit(self, dataset: pd.DataFrame, labels: pd.Series, ignore_general: bool = False, seed: int | None = None) -> None:
        # Set seed for regularity in testing
        if seed is not None:
            set_seed(seed)

        if self.general_case is None:
            # Asign a general case based on labels
            values, counts = np.unique(labels, return_counts=True)
            mode_label = values[np.argmax(counts)]
            self.general_case = mode_label if type(
                mode_label) is str else str(mode_label)
            self._root.set_general_case(self.general_case)

        # build the mask to isolate cornerstone cases
        if ignore_general:
            mask = labels.values.astype(str) != self.general_case
            dataset_cs: pd.DataFrame = dataset[mask].astype(bool)
            labels_cs: pd.Series = labels[mask]
        else:
            dataset_cs = dataset.astype(bool)
            labels_cs = labels

        # build the knowledge tree
        for data, label in zip(dataset_cs.values, labels_cs.values):
            features = [dataset_cs.columns[i] for i in np.where(data)[0]]
            self._root.ripple_down(features, label)

    def predict(self, dataset: pd.DataFrame) -> pd.Series:
        predictions = []
        # Iterate over each row in the dataset
        for data in dataset.astype(bool).values:
            # predict the conclusion of the row
            features = [dataset.columns[i] for i in np.where(data)[0]]
            conclusion = self._root.predict(features)
            # append the conclusion to the predictions
            predictions.append(
                conclusion if conclusion is not None else self.general_case)
        return pd.Series(data=predictions)

    def add_new_cornerstone(self, data, new_conclusion, new_prerequisites=None) -> None:
        # Add the new cornerstone to the RDR tree
        self._root.ripple_down(data,
                               new_conclusion, new_prerequisites)

    def __str__(self) -> str:
        return f"General Case: {self.general_case}\n" + self._root.visualize()

    class RDRNode:
        def __init__(self, prerequisites: list | None, conclusion: str | None) -> None:
            """
            prerequisites: list of features that must be met before this rule can be applied
            conclusion: the conclusion of this rule
            left: the left child of this rule, applies when the prerequisites are not met
            right: the right child of this rule, applies when the prerequisites are met
            """
            self._prerequisites: list | None = prerequisites
            self._conclusion: str | None = conclusion if type(
                conclusion) is str or conclusion is None else str(conclusion)
            self._left: RDR.RDRNode | None = None
            self._right: RDR.RDRNode | None = None

        def set_general_case(self, general_case: str) -> None:
            self._general_case = general_case

        def _add_node(self, features, node) -> None:
            # If the node's prerequisites are met
            if self.is_fulfilled(features):
                if self._right is None:
                    self._right = node
                else:
                    self._right._add_node(features, node)
            else:
                if self._left is None:
                    self._left = node
                else:
                    self._left._add_node(features, node)

        def is_fulfilled(self, features) -> bool:
            # return true if all the prerequisites are met
            if self._prerequisites is not None:
                for prerequisite in self._prerequisites:
                    if prerequisite not in features:
                        return False
            return True

        def _traverse_to_conclusion_leaf(self, features) -> RDR.RDRNode | None:
            # Traverse to the conclusion leaf
            if self.is_fulfilled(features):
                conclusion_node = self
                if self._right is not None:
                    right_conclusion = self._right._traverse_to_conclusion_leaf(features)
                    if right_conclusion is not None:
                        conclusion_node = right_conclusion
                return conclusion_node
            elif self._left is not None:
                return self._left._traverse_to_conclusion_leaf(features)
            return None

        def _contradict(self, features, new_conclusion, new_prerequisites=None) -> None:
            # If there are no new prerequisites, generate some
            if new_prerequisites is None:
                # If there are prerequisites, use them as a base
                if self._prerequisites is not None:
                    new_prerequisites = [p for p in self._prerequisites]
                    possible_features = [f for f in features if f not in new_prerequisites]
                    if len(possible_features) != 0:
                        # append the random feature that is not already in the prerequisites
                        new_prerequisites.append(choice(possible_features))
                    else:
                        # Didn't find a new feature to append, so remove the feature instead
                        duplicate_features = [f for f in features if f in new_prerequisites]
                        # remove the random feature that is already in the prerequisites
                        new_prerequisites.remove(choice(duplicate_features))
                else:
                    # Append Random Feature
                    new_prerequisites = []
                    new_prerequisites.append(choice(features))

            # Add the new cornerstone to the RDR tree
            self._add_node(features, RDR.RDRNode(new_prerequisites, new_conclusion))

        def _manifest(self, features, new_conclusion, new_prerequisites=None) -> None:
            # If there are no new prerequisites, generate some
            if new_prerequisites is None:
                # Since this is a manifestation, the new prerequisites should be the features
                # This is because we triggered the else branch, thus prerequisites are not met
                # Can choose any feature to be the new prerequisite as long as prerequisites are not met
                # Easier Just to use a random feature from the features list that is not already in the prerequisites
                new_prerequisites = []
                possible_features = [f for f in features if f not in new_prerequisites]
                if len(possible_features) != 0:
                    # append the random feature that is not already in the prerequisites
                    new_prerequisites.append(choice(possible_features))
                else:
                    # all features are already in the prerequisites, but prerequisites were not met, features lacking
                    # in this case, just use all the features as the prerequisites
                    new_prerequisites = [f for f in features]

            self._add_node(features, RDR.RDRNode(new_prerequisites, new_conclusion))


        def ripple_down(self, features, label, new_prerequisites = None) -> str | None:
            # Get the conclusion of this case
            conclusion = self.predict(features)
            # If it contradicts the label, add right cornerstone
            if conclusion is not None and conclusion != label:
                # Create a new cornerstone with the label as the conclusion
                conclusion_leaf = self._traverse_to_conclusion_leaf(features)
                if conclusion_leaf is not None:
                    conclusion_leaf._contradict(
                        features, label, new_prerequisites)
                else:
                    # For the general case, i.e. root rule
                    self._contradict(
                        features, label, new_prerequisites)
            # The rule did not activate the label, so add Left Cornerstone
            elif conclusion is None and label != self._general_case:
                # Create a new cornerstone with the label as the conclusion
                first_empty_left = self._traverse_to_conclusion_leaf(features)
                if first_empty_left is not None:
                    first_empty_left._manifest(
                        features, label, new_prerequisites)
                else:
                    # For the general case, i.e. root rule, this is the first rule, so contradict instead
                    self._contradict(
                        features, label, new_prerequisites)
                

        def predict(self, features) -> str | None:
            # If prerequisite is met
            if self.is_fulfilled(features):
                # Set the last conclusion to this rule's conclusion
                last_conclusion = self._conclusion
                # If the rule has a right child
                if self._right is not None:
                    # calculate the right child's prediction, this should be the conclusion of the right child
                    right_conclusion = self._right.predict(features)
                    # if the right child has a conclusion
                    if right_conclusion is not None:
                        # set the last conclusion to the right child's conclusion
                        last_conclusion = right_conclusion
                return last_conclusion
            # If prerequisite is not met, but has a left child
            elif self._left is not None:
                # calculate the left child's prediction, this should be the conclusion of the left child
                return self._left.predict(features)
            # Default of no conclusion
            return None
        
        def visualize(self, depth: int = 0, side = None) -> str:
            # Create the string to return
            string = ' ' * depth + f"{side}" if side is not None else ''
            # Add the current node's visualization to the string
            string += self.__str__()
            # If there is a left child
            if self._left is not None:
                # Add the left child's visualization to the string
                string += self._left.visualize(depth + 1, "L: ")
            # If there is a right child
            if self._right is not None:
                # Add the right child's visualization to the string
                string += self._right.visualize(depth + 1, "R: ")
            return string

        def __str__(self) -> str:
            # Prerequisites => Conclusion
            return f"{' & '.join(self._prerequisites) if self._prerequisites is not None else 'Any Feature'} => {self._conclusion if self._conclusion is not None else 'General Case'}\n"

if __name__ == '__main__':
    df = pd.read_csv('./test/diabetes_012_health_indicators_BRFSS2015.csv').drop(columns=['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income'], axis=1)
    labels = df['Diabetes_012'].replace(
        {0.0: 'Normal', 1.0: 'Pre-diabetes', 2.0: 'Diabetes'}).head(90000)
    test_labels = df['Diabetes_012'].replace(
        {0.0: 'Normal', 1.0: 'Pre-diabetes', 2.0: 'Diabetes'}).tail(10000)
    test_df = df.drop(columns=['Diabetes_012'], axis=1).tail(10000)
    df = df.drop(columns=['Diabetes_012'], axis=1).head(90000)
    rdr = RDR()
    rdr.fit(df, labels, seed=42)
    print(rdr)
    pred = rdr.predict(test_df)
    tp = 0
    for p, l in zip(pred.values, test_labels):
        if p == l:
            tp += 1
    print(tp / len(pred))
