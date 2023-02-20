#ifndef KD_TREE_H
#define KD_TREE_H
#include "include.h"
#include "array.h"

typedef struct kd_node_s
{
    float *pos;
    void *data;
    int dir;
    struct kd_node_s *left, *right;
} kd_node_s;
typedef struct kd_node_s *kd_node;

typedef struct kd_rect_s
{
    float *min, *max;
} kd_rect_s;
typedef struct kd_rect_s *kd_rect;

typedef struct kd_tree_s
{
    int dim;
    kd_node root;
    kd_rect rect;
} kd_tree_s;
typedef struct kd_tree_s *kd_tree;

/* create a kd-tree for "k"-dimensional data */
kd_tree kd_create(int k);

/* remove all the elements from the tree */
void kd_clear(kd_tree tree);

/* free the struct kdtree */
void kd_free(kd_tree tree);

/* build a tree from an array of positions */
kd_tree kd_build(float *pos, int len, int dim, void *data, size_t size);

/* insert a node, specifying its position */
void kd_insert(kd_tree tree, float *pos, void *data);

/* find a minimum / maximum along the axis */
kd_node kd_find_min(kd_tree tree, int axis);
kd_node kd_find_max(kd_tree tree, int axis);

/* delete a point from the tree */
void kd_delete(kd_tree tree, float *pos);

/* print a tree */
void kd_print(FILE *stream, kd_tree tree);

typedef struct kd_query_s
{
    kd_node node;
    float dist;
} kd_query_s;
typedef struct kd_query_s *kd_query;

/* find a nearest neighbour */
kd_query kd_find_nearest(kd_tree tree, float *pos);

typedef struct query_stack_s
{
    kd_query query;
    struct query_stack_s *next;
} query_stack_s;
typedef struct query_stack_s *query_stack;

void stack_push(query_stack *stack, kd_query query);
kd_query stack_pop(query_stack *stack);
void free_stack(query_stack stack);

/* find a nearest neighbours in a range */
query_stack kd_find_range(kd_tree tree, float *pos, float range);

#endif