// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main extends DataGenerator {

    public static void main(String[] args) {
        int n_tasks = 7;
        int n_machines = 25;
        int iteration = 0;
        String outputDir = "training_data_classification";
        while (true) {
            System.out.println(iteration);
            iteration++;
            int[][] workingTimeMatrix = generateRandomMatrix(n_tasks, n_machines);
            Tree tree = new Tree(n_tasks, n_machines, workingTimeMatrix);
            Node root = new Node(workingTimeMatrix);
            Node bestNode = tree.branchAndBound(root);
            for (int tasks = -1; tasks < n_tasks; tasks++) {
                int[][] timeMatrix = new int[0][0];
                for (int n_tasksChosen = tasks + 1; n_tasksChosen < n_tasks; n_tasksChosen++) {
                    timeMatrix = Node.appendArrayToMatrix(timeMatrix, workingTimeMatrix[bestNode.tasks[n_tasksChosen]]);
                }
                if (timeMatrix.length == 0)
                    break;
                if (tasks == -1) {
                    saveToTxtFile(new int[n_machines], timeMatrix, bestNode.getValue(), outputDir + "/" + timeMatrix.length + "_" + n_machines + ".txt");
                } else {
                    saveToTxtFile(bestNode.getState()[tasks], timeMatrix, bestNode.getValue(), outputDir + "/" + timeMatrix.length + "_" + n_machines + ".txt");
                }
            }
        }
    }
}