#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  /*Fetch data from pointer*/
  uint32_t arows = a_matrix->rows;
  uint32_t acols = a_matrix->cols;
  int32_t *adata = a_matrix->data;
  uint32_t brows = b_matrix->rows;
  uint32_t bcols = b_matrix->cols;
  int32_t *bdata = b_matrix->data;

  // First flip b

  /*The vertical flip*/
  uint32_t swap_times_vertical = brows / 2;
  for (unsigned int i = 0; i < swap_times_vertical; i ++) {
    for (unsigned int j = 0; j < bcols; j ++) {
      uint32_t x = i * bcols + j;
      uint32_t y = (brows - i - 1) * bcols + j;
      int32_t tmp = bdata[y];
      bdata[y] = bdata[x];
      bdata[x] = tmp;
    }
  }

  /*Horizontal flip*/
  uint32_t swap_times_horizontal = bcols / 2;
  for (unsigned int i = 0; i < swap_times_horizontal; i++) {
    for (unsigned int j = 0; j < brows; j ++) {
      uint32_t x = j * bcols + i;
      uint32_t y = j * bcols + bcols - i - 1;
      int32_t tmp = bdata[y];
      bdata[y] = bdata[x];
      bdata[x] = tmp;
    }
  }
  /*Let's cal the row and col of the output*/
  uint32_t orows = arows - brows + 1;
  uint32_t ocols = acols - bcols + 1;
  matrix_t * output = malloc(sizeof(matrix_t));
  if (output == NULL) {
    return -1;
  }
  output->cols = ocols;
  output->rows = orows;
  output->data = malloc(sizeof(int32_t) * ocols * orows);
  if (output->data == NULL) {
    free(output);
    return -1;
  }
  /*Initialize the matrix*/
  for (unsigned int i = 0; i < orows; i++) {
      for (unsigned int j = 0; j < ocols; j ++) {
        output->data[j + ocols * i] = 0;
      }
  }

  // Then start dot mul
  /*  Every time a has a start;
    then horizontally move b's cols num ;
    and meanwhile make b's row num times move;
    each move a's row num distance */ 
    for (unsigned int i = 0; i < orows; i++) {
      for (unsigned int j = 0; j < ocols; j++) {
        
        /*Above is how many dot mul we op*/
        /*Below is each dot mul we gonna loop x and y*/
        int32_t sum = 0;
        int32_t b_index = 0;
        for (unsigned int x = i; x < i + brows; x ++) {
          for (unsigned int y = j; y < j + bcols; y ++) {
            sum += adata[y + acols * x] * bdata[b_index];
            b_index += 1;
          }
        }
        output->data[i * ocols + j] = sum;
      }
    }
  *output_matrix = output;
  return 0;
}


// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
