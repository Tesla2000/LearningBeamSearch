//// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
//// then press Enter. You can now see whitespace characters in your code.
//public class GenValidationData extends DataGenerator{
//
//    public static void main(String[] args) {
//        int n_tasks = 10;
//        int n_machines = 25;
//        int i = 0;
//        while (true) {
//            System.out.println(i);
//            i++;
//            int[][] workingTimeMatrix = generateRandomMatrix(n_tasks, n_machines);
//            Tree tree = new Tree(n_tasks, n_machines, workingTimeMatrix);
//            Node root = new Node(workingTimeMatrix);
//            Node bestNode = tree.branchAndBound(root);
//            int tasks = 0;
//            int[][] timeMatrix = new int[0][0];
//            for (int n_tasksChosen = tasks; n_tasksChosen < n_tasks; n_tasksChosen++) {
//                timeMatrix = Node.appendArrayToMatrix(timeMatrix, workingTimeMatrix[bestNode.tasks[n_tasksChosen]]);
//            }
//            saveToTxtFile(new int[n_machines], timeMatrix, bestNode.getValue(), "validation_data/" + timeMatrix.length + "_" + n_machines + ".txt");
//        }
//    }
//}