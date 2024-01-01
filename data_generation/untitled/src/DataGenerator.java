import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class DataGenerator {


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
            int minimum = Integer.MAX_VALUE;
            for (int b : previousState) {
                minimum = Math.min(minimum, b);
            }
            for (int b : previousState) {
                writer.write((b - minimum) + " ");
            }
            writer.newLine();
            for (int b : flattenedArray) {
                writer.write((char) b);
            }
            writer.write(String.valueOf(uint32Value - minimum));
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
