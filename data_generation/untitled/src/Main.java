import java.util.Random;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {
    static int n_tasks = 7;
    static int m_machines = 10;
    static int[][] workingTimeMatrix = generateRandomMatrix(n_tasks, m_machines);

    public static void main(String[] args) {
        Tree tree = new Tree(n_tasks, m_machines, workingTimeMatrix);
        Node root = new Node(workingTimeMatrix);
        Node bestNode = tree.branchAndBound(root);
//        displayMatrix(workingTimeMatrix);
//        System.out.println();
//        displayMatrix(bestNode.getState());
    }

    public static int[][] generateRandomMatrix(int rows, int columns) {
        int[][] matrix = new int[rows][columns];
        Random rand = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                // Generate random integers in the range 1 to 99
                matrix[i][j] = rand.nextInt(99) + 1;
            }
        }

        return matrix;
    }

    public static void displayMatrix(int[][] matrix) {
        for (int[] row : matrix) {
            for (int value : row) {
                System.out.printf("%4d", value); // Adjust formatting as needed
            }
            System.out.println();
        }
    }
}