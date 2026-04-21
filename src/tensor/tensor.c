#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int ndim;
  int* shape; /* Shape is an array that holds the number of elements in each dimension */
  int* stride; /*  Stride is the amount by which we will move to aquire +1 in wanted dimension */
  float* data;
} TensorBase;

TensorBase* create_tensor_b(int ndim, int* shape) {
  TensorBase* t = (TensorBase*) malloc(sizeof(TensorBase));
  if (!t) return NULL; /* I will impl safe malloc later, I shall always check if malloc returned me memory location and not NULL */
  t->ndim = ndim;
  t->shape = (int*) malloc(ndim * sizeof(int));
  memcpy(t->shape, shape, ndim * sizeof(int));
  t->stride = (int*) malloc(ndim * sizeof(int));
  t->stride[ndim - 1] = 1;
  for(int i = ndim - 1; i > 0; i--) {
    t->stride[i-1] = t->shape[i] * t->stride[i]; 
  }
  size_t total = 1;
  for(int i = 0; i < ndim; i++){
    total *= (size_t) t->shape[i];
  }
  t->data = (float*) malloc(total * sizeof(float));
  return t;
}

void destruct_tensor_b(TensorBase* t) {
  if (!t) return;
  free(t->data);
  free(t->shape);
  free(t->stride);
  free(t);
}

float get(TensorBase* t, int* index) {
 size_t offset = 0;
 for(int i = 0; i < t->ndim; i++) {
    /* if (index[i] < 0 || index[i] >= t->shape[i]) I shell have check of this type, I shall also sheck if t and index are not NULL*/
    offset += (size_t) t->stride[i] * index[i];
 }
 return t->data[offset];
}

void set(TensorBase* t, int* index, float val) {
  size_t offset = 0;
  for(int i = 0; i < t->ndim; i++) {
    offset += (size_t) t->stride[i] * index[i];
  }
  t->data[offset] = val;
}