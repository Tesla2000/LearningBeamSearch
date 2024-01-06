public class Tree {
    int n_tasks;
    int m_machines;
    int bestValue = Integer.MAX_VALUE;
    int[][] workingTimeMatrix;
    Node bestNode = null;

    public Tree(int n_tasks, int m_machines, int[][] workingTimeMatrix) {
        this.n_tasks = n_tasks;
        this.m_machines = m_machines;
        this.workingTimeMatrix = workingTimeMatrix;
    }

    public Node branchAndBound(Node root) {
        innerBranchAndBound(root);
        return bestNode;
    }

    private void innerBranchAndBound(Node root) {
        if (root.tasks.length == n_tasks) {
            if (root.getValue() < bestValue){
                bestValue = root.getValue();
                bestNode = root;
            }
            return;
        }
        for (int task = 0; task < n_tasks; task++) {
            if (isValueInArray(root.tasks, task))
                continue;
            Node childNode = new Node(root, task);
            root.children.add(childNode);
            int bestPossibleValue = childNode.getValue();
            childNode.bestValueAtCreation = bestValue;
            for (int not_done_task = 0; not_done_task < n_tasks; not_done_task++) {
                if (!isValueInArray(childNode.tasks, not_done_task)) {
                    bestPossibleValue += workingTimeMatrix[not_done_task][workingTimeMatrix[0].length - 1];
                }
            }
            if (bestPossibleValue < bestValue) {
                innerBranchAndBound(childNode);
            }
        }
    }

    public static boolean isValueInArray(int[] array, int value) {
        for (int element : array) {
            if (element == value) {
                return true;
            }
        }
        return false;
    }
}