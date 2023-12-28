import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {

    public static void main(String[] args) {
        int n_tasks = 7;
        int n_machines = 10;
        for (int i = 0; i < 100; i++) {
            System.out.println(i);
            int[][] workingTimeMatrix = generateRandomMatrix(n_tasks, n_machines);
            Tree tree = new Tree(n_tasks, n_machines, workingTimeMatrix);
            Node root = new Node(workingTimeMatrix);
            Node bestNode = tree.branchAndBound(root);
            for (int tasks = 0; tasks < n_tasks; tasks++) {
                int[][] timeMatrix = new int[0][0];
                for (int n_tasksChosen = tasks; n_tasksChosen < n_tasks; n_tasksChosen++) {
                    timeMatrix = Node.appendArrayToMatrix(timeMatrix, workingTimeMatrix[bestNode.tasks[n_tasksChosen]]);
                }
                saveToTxtFile(bestNode.getState()[tasks], flattenMatrix(timeMatrix), bestNode.getValue(), "data/" + timeMatrix.length + "_" + n_machines + ".txt");
            }

        }

    }

    public static int[][] generateRandomMatrix(int rows, int columns) {
        int[][] matrix = new int[rows][columns];
        Random rand = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                // Generate random integers in the range 1 to 99
                matrix[i][j] = rand.nextInt(255) + 1;
            }
        }

        return matrix;
    }

    public static int[] flattenMatrix(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int totalElements = rows * cols;
        int[] flattenedArray = new int[totalElements];

        int index = 0;
        for (int[] row : matrix) {
            for (int value : row) {
                flattenedArray[index++] = value;
            }
        }
        return flattenedArray;
    }

    public static void saveToTxtFile(int[] previousState, int[] flattenedArray, int uint32Value, String fileName) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName, true))) {
            for (int b : previousState) {
                writer.write(b + " ");
            }
            writer.newLine();
            for (int b : flattenedArray) {
                writer.write((char) b);
            }
            writer.write(String.valueOf(uint32Value));
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}