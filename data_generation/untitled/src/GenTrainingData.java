import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class GenTrainingData extends DataGenerator{

    public static void main(String[] args) {
        int n_tasks = 7;
        int n_machines = 25;
        int i = 0;
        while (true) {
            System.out.println(i);
            i++;
            int[][] workingTimeMatrix = generateRandomMatrix(n_tasks, n_machines);
            Tree tree = new Tree(n_tasks, n_machines, workingTimeMatrix);
            Node root = new Node(workingTimeMatrix);
            Node bestNode = tree.branchAndBound(root);
            saveToTxtFile(workingTimeMatrix, bestNode.tasks, "training_data/" + workingTimeMatrix.length + "_" + n_machines + ".txt");
        }
    }
    public static void saveToTxtFile(int[][] array, int[] tasks, String fileName) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName, true))) {
            for (int[] line : array) {
                for (int b : line)
                    writer.write(b + " ");
                writer.newLine();
            }
            for (int b : tasks) {
                writer.write(b + " ");
            }
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}