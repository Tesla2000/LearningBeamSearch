import java.util.ArrayList;
import java.util.Random;

public class Main extends DataGenerator {
    static Random random = new Random();
    static int[] once = new int[10];
    static int[] zeros = new int[10];

    public static void main(String[] args) {
        int n_tasks = 9;
        int m_machines = 25;
        int iteration = 0;
        OutputPath outputPath = OutputPath.CLASSIFICATION;
        String outputDir = null;
        if (outputPath == OutputPath.CLASSIFICATION){
            outputDir = "training_data_classification";
        } else if (outputPath == OutputPath.REGRESSION) {
            outputDir = "training_data_regression";
        }
        while (true) {
            System.out.println(iteration);
            iteration++;
            int[][] workingTimeMatrix = generateRandomMatrix(n_tasks, m_machines);
            Tree tree = new Tree(n_tasks, m_machines, workingTimeMatrix);
            Node root = new Node(workingTimeMatrix);
            Node bestNode = tree.branchAndBound(root);
            if (outputPath == OutputPath.CLASSIFICATION){
                ArrayList<Node> correctPath = new ArrayList<>();
                while (bestNode != null) {
                    correctPath.add(bestNode);
                    bestNode = bestNode.parent;
                }
                saveClassification(n_tasks, m_machines, workingTimeMatrix, root, outputDir, correctPath, 5);
            } else if (outputPath == OutputPath.REGRESSION) {
                saveRegression(n_tasks, m_machines, workingTimeMatrix, bestNode, outputDir);
            }

        }
    }

    public static void saveRegression(int n_tasks, int m_machines, int[][] workingTimeMatrix, Node bestNode, String outputDir) {
        for (int tasks = -1; tasks < n_tasks; tasks++) {
            int[][] timeMatrix = new int[0][0];
            for (int n_tasksChosen = tasks + 1; n_tasksChosen < n_tasks; n_tasksChosen++) {
                timeMatrix = Node.appendArrayToMatrix(timeMatrix, workingTimeMatrix[bestNode.tasks[n_tasksChosen]]);
            }
            if (timeMatrix.length == 0)
                break;
            if (timeMatrix.length < 3)
                return;
            if (tasks == -1) {
                saveToTxtFile(new int[m_machines], timeMatrix, bestNode.getValue(), outputDir + "/" + timeMatrix.length + "_" + m_machines + ".txt");
            } else {
                saveToTxtFile(bestNode.getState()[tasks], timeMatrix, bestNode.getValue(), outputDir + "/" + timeMatrix.length + "_" + m_machines + ".txt");
            }
        }
    }

    public static void saveClassification(int n_tasks, int m_machines, int[][] workingTimeMatrix, Node root, String outputDir, ArrayList<Node> correctPath, float zerosRate) {
        Node node = root;
        int tasks = node.getState().length-1;
        if (tasks != -1) {
            int[][] timeMatrix = new int[0][0];
            for (int task = 0; task < n_tasks; task++) {
                if (Tree.isValueInArray(node.tasks, task))
                    continue;
                timeMatrix = Node.appendArrayToMatrix(timeMatrix, workingTimeMatrix[task]);
            }
            if (timeMatrix.length != 0) {
                if (correctPath.contains(node)) {
                    saveToTxtFile(node.getState()[tasks], timeMatrix, node.bestValueAtCreation, outputDir + "/" + 1 + "_" + timeMatrix.length + "_" + m_machines + ".txt");
                    once[tasks]++;
                }
                if (random.nextDouble() < 1.0 / tasks / tasks && once[tasks]*zerosRate > zeros[tasks]) {
                    saveToTxtFile(node.getState()[tasks], timeMatrix, node.bestValueAtCreation, outputDir + "/" + 0 + "_" + timeMatrix.length + "_" + m_machines + ".txt");
                    zeros[tasks]++;
                }
            }
        }
        for (Node childNode: node.children) {
            saveClassification(n_tasks, m_machines, workingTimeMatrix, childNode, outputDir, correctPath, zerosRate);
        }
    }
}