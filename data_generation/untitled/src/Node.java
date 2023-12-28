import java.util.Arrays;

public class Node {
    private int value;
    int[][] state;
    int[][] workingTimeMatrix;
    Node parent;
    int[] tasks;

    public Node(Node parent, int task) {
        this.workingTimeMatrix = parent.workingTimeMatrix;
        this.parent = parent;
        this.tasks = appendValue(parent.tasks, task);
        value = 0;
    }

    public Node(int[][] workingTimeMatrix) {
        this.workingTimeMatrix = workingTimeMatrix;
        this.parent = null;
        this.tasks = new int[0];
        this.state = new int[0][0];
        value = 0;
    }

    public int getValue() {
        if (parent == null)
            return 0;
        if (value != 0)
            return value;

        getState();
        return value;
    }

    public int[][] getState() {
        if (state != null)
            return state;
        state = appendArrayToMatrix(parent.getState(), workingTimeMatrix[tasks[tasks.length - 1]]);
        if (state.length == 1) {
            for (int i=1; i < state[0].length; i++)
                state[0][i] += state[0][i - 1];
            return state;
        }
        state[state.length - 1][0] += state[state.length - 2][0];
        for (int i=1; i < state[0].length; i++)
            state[state.length - 1][i] += Math.max(state[state.length - 2][i], state[state.length - 1][i - 1]);
        value = state[state.length - 1][state[0].length -1];
        return state;
    }

    public static int[] appendValue(int[] array, int value) {
        int originalLength = array.length;
        array = Arrays.copyOf(array, originalLength + 1);

        array[originalLength] = value;

        return array;
    }

    public static int[][] appendArrayToMatrix(int[][] matrix, int[] array) {
        int rows = matrix.length;
        int columns = array.length;
        int[][] newMatrix = new int[rows + 1][columns]; // Create a new matrix with an additional row

        // Copy the original matrix to the new matrix
        for (int i = 0; i < rows; i++) {
            newMatrix[i] = Arrays.copyOf(matrix[i], columns);
        }

        // Append the new array to the new matrix as the last row
        newMatrix[rows] = Arrays.copyOf(array, columns);

        return newMatrix;
    }
}
