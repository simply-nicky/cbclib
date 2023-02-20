#include "kd_tree.h"

// Minimum of three points along the axis
static kd_node min_node(kd_node a, kd_node b, kd_node c, int axis)
{
    kd_node res = a;
    if (b && b->pos[axis] < res->pos[axis]) res = b;
    if (c && c->pos[axis] < res->pos[axis]) res = c;
    return res;
}

static kd_node find_min(kd_node root, int axis)
{
    // fell out of tree
    if (!root) return NULL;

    // axis == splitting axis
    if (root->dir == axis)
    {
        if (!root->left) return root;
        else return find_min(root->left, axis);
    }
    else
    {
        // else minimum of both children and root
        return min_node(root, find_min(root->left, axis), find_min(root->right, axis), axis);
    }
}

// Maximum of three points along the axis
static kd_node max_node(kd_node a, kd_node b, kd_node c, int axis)
{
    kd_node res = a;
    if (b && b->pos[axis] > res->pos[axis]) res = b;
    if (c && c->pos[axis] > res->pos[axis]) res = c;
    return res;
}

static kd_node find_max(kd_node root, int axis)
{
    // fell out of tree
    if (!root) return NULL;

    // axis == splitting axis
    if (root->dir == axis)
    {
        if (!root->right) return root;
        else return find_max(root->right, axis);
    }
    else
    {
        // else maximum of both children and root
        return max_node(root, find_max(root->left, axis), find_max(root->right, axis), axis);
    }
}

static kd_rect new_rect(float *min, float *max, int dim)
{
    kd_rect rect = malloc(sizeof(struct kd_rect_s));
    rect->min = MALLOC(float, dim);
    rect->max = MALLOC(float, dim);
    memcpy(rect->min, min, dim * sizeof(float));
    memcpy(rect->max, max, dim * sizeof(float));
    return rect;
}

static void init_rect(kd_node root, kd_rect rect, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        rect->min[i] = find_min(root, i)->pos[i];
        rect->max[i] = find_max(root, i)->pos[i];
    }
}

static void free_rect(kd_rect rect)
{
    DEALLOC(rect->min); DEALLOC(rect->max); DEALLOC(rect);
}

kd_tree kd_create(int k)
{
    kd_tree tree = malloc(sizeof(struct kd_tree_s));
    tree->dim = k;
    tree->root = NULL;
    tree->rect = NULL;

    return tree;
}

static void clear_node(kd_node node)
{
    if (node)
    {
        clear_node(node->left);
        clear_node(node->right);
        DEALLOC(node);
    }
}

void kd_clear(kd_tree tree)
{
    clear_node(tree->root);
    if (tree->rect) DEALLOC(tree->rect);
}

void kd_free(kd_tree tree)
{
    if (tree)
    {
        kd_clear(tree);
        DEALLOC(tree);
    }
}

static kd_node kd_create_node(float *pos, int dir, void *data)
{
    kd_node node = malloc(sizeof(struct kd_node_s));
    node->pos = pos;
    node->dir = dir;
    node->data = data;
    node->left = node->right = NULL;
    return node;
}

static int compare_node(const void *a, const void *b, void *arg)
{
    int dir = *(int *)arg;
    if (*((*(kd_node *)a)->pos + dir) > *((*(kd_node *)b)->pos + dir)) return 1;
    else if (*((*(kd_node *)a)->pos + dir) < *((*(kd_node *)b)->pos + dir)) return -1;
    else return 0;
}

static kd_node kd_build_node(kd_node *nodes, int l, int r, int dir, int dim)
{
    kd_node node;
    
    // Empty node if empty partition
    if (r <= l) node = NULL;
    // Creating leaf node
    else if (r == l + 1)
    {
        // node = kd_create_node(n + l * dim, dir);
        node = nodes[l]; node->dir = dir;
    }
    else
    {
        // Sorting the coordinates 'pos' along the dimension 'dir'
        kd_node *median = wirthmedian_r(nodes + l, r - l, sizeof(kd_node), compare_node, &dir);
        int k = median - nodes;

        // Populating the node
        node = *median; node->dir = dir;
        node->left = kd_build_node(nodes, l, k, (dir + 1) % dim, dim);
        node->right = kd_build_node(nodes, k + 1, r, (dir + 1) % dim, dim);
    }
    
    return node;
}

kd_tree kd_build(float *pos, int len, int dim, void *data, size_t size)
{
    kd_tree tree = kd_create(dim);
    kd_node *nodes = MALLOC(kd_node, len);
    for (int i = 0; i < len; i++) nodes[i] = kd_create_node(pos + i * dim, 0, data + i * size);
    tree->root = kd_build_node(nodes, 0, len, 0, dim);
    DEALLOC(nodes);

    if (!(tree->rect)) tree->rect = new_rect(pos, pos, tree->dim);
    init_rect(tree->root, tree->rect, tree->dim);
    return tree;
}

static kd_node insert_node(kd_node root, float *pos, int dir, int dim, void *data)
{
    if (!root)
    {
        // Create new node if empty
        return kd_create_node(pos, dir, data);
    }
    else if (pos[root->dir] == root->pos[root->dir])
    {
        ERROR("insert_node: inserting a duplicate point.");
    }
    else if (pos[root->dir] < root->pos[root->dir])
    {
        // left of splitting line
        root->left = insert_node(root->left, pos, (root->dir + 1) % dim, dim, data);
    }
    else
    {
        // on or right of splitting line
        root->right = insert_node(root->right, pos, (root->dir + 1) % dim, dim, data);
    }
    return root;
}

void kd_insert(kd_tree tree, float *pos, void *data)
{
    tree->root = insert_node(tree->root, pos, 0, tree->dim, data);

    if (!(tree->rect)) tree->rect = new_rect(pos, pos, tree->dim);
    else
    {
        for (int i = 0; i < tree->dim; i++)
        {
            if (pos[i] < tree->rect->min[i]) tree->rect->min[i] = pos[i];
            if (pos[i] > tree->rect->max[i]) tree->rect->max[i] = pos[i];
        }
    }
}

kd_node kd_find_min(kd_tree tree, int axis)
{
    return find_min(tree->root, axis);
}

kd_node kd_find_max(kd_tree tree, int axis)
{
    return find_max(tree->root, axis);
}

static int equal_points(const float *a, const float *b, int dim)
{
    for (int i = 0; i < dim; i++) {if (a[i] != b[i]) return 0;}
    return 1;
}

static kd_node kd_delete_node(kd_node root, float *pos, int dim)
{
    if (!root) return NULL;

    // Found the node
    if (equal_points(root->pos, pos, dim))
    {
        kd_node node;
        // Take replacement from right
        if (root->right)
        {
            /* Swapping the nodes */
            node = find_min(root->right, root->dir);
            root->pos = node->pos; root->data = node->data;

            root->right = kd_delete_node(root->right, root->pos, dim);
        }
        // Take replacement from left
        else if (root->left)
        {
            /* Swapping the nodes */
            node = find_min(root->left, root->dir);
            root->pos = node->pos; root->data = node->data;

            root->right = kd_delete_node(root->left, root->pos, dim);
            root->left = NULL;
        }
        else DEALLOC(root);
    }
    // Search left subtree
    else if (pos[root->dir] < root->pos[root->dir])
    {
        root->left = kd_delete_node(root->left, pos, dim);
    }
    // Search right subtree
    else root->right = kd_delete_node(root->right, pos, dim);

    return root;
}

void kd_delete(kd_tree tree, float *pos)
{
    tree->root = kd_delete_node(tree->root, pos, tree->dim);
    if (tree->rect)
    {
        if (tree->root)
        {
            for (int i = 0; i < tree->dim; i++)
            {
                if (pos[i] == tree->rect->min[i])
                {
                    tree->rect->min[i] = find_min(tree->root, i)->pos[i];
                }
                if (pos[i] == tree->rect->max[i])
                {
                    tree->rect->max[i] = find_max(tree->root, i)->pos[i];
                }
            }
        }
        else free_rect(tree->rect);
    }
}

/* prints a node, with its key and prefix. */
static void print_node(FILE *stream, kd_node node, int dim, char *full_buf)
{
    /* the first string is the fixed part that prefix the whole
     * subtree, the second is the node's name. */
    if (node->left || node->right) fprintf(stream, "%s+-+(", full_buf);
    else fprintf(stream, "%s+--(", full_buf);
    for (int i = 0; i < dim; i++) fprintf(stream, "%.2f ", node->pos[i]);
    fprintf(stream, ")\n");
}

/* prints a subtree (the first one in the sequence)
 * printing a left or right subtree is controlled by the
 * prf_right and prf_left parameters passed to the call. */
static void kd_print_node(FILE *stream, kd_node root, int dim, const char *prf_right, const char *prf_left,
                          char *buf, int buf_sz, char *full_buf)
{
    if (root->right)
    {
        /* add the prefix for the right subtree, this is prf_right for
         * the right part (the first, before the root node) */
        int res = snprintf(buf, buf_sz, "%s", prf_right);
        kd_print_node(stream, root->right, dim, "  ", "| ", buf + res, buf_sz - res, full_buf);
        *buf = '\0';
    }
    print_node(stream, root, dim, full_buf);
    if (root->left)
    {
        /* add the prefix for the left subtree, this is prf_left
         * for the left part (the second, after the root node) */
        int res = snprintf(buf, buf_sz, "%s", prf_left);
        kd_print_node(stream, root->left, dim, "| ", "  ", buf + res, buf_sz - res, full_buf);
        *buf = '\0';
    }
}

void kd_print(FILE *stream, kd_tree tree)
{
    if (tree->root)
    {
        char buffer[1024];
        kd_print_node(stream, tree->root, tree->dim, "  ", "  ", buffer, sizeof(buffer), buffer);
    }
    else fprintf(stream, "empty tree\n");
}

static float distance_to_rect(float *pos, kd_rect rect, int dim)
{
    float dist = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        if (pos[i] < rect->min[i]) dist += SQ(rect->min[i] - pos[i]);
        if (pos[i] > rect->max[i]) dist += SQ(pos[i] - rect->max[i]);
    }
    return dist;
}

static kd_query new_query(kd_node node, float dist)
{
    kd_query query = malloc(sizeof(struct kd_query_s));
    query->node = node; query->dist = dist;
    return query;
}

static kd_query find_nearest(kd_node root, float *pos, int dim, kd_rect rect, kd_query close)
{
    // Fell out of tree
    if (!root) return close;
    // This cell is too far away
    if (distance_to_rect(pos, rect, dim) >= close->dist) return close;

    // Update if the root is closer
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) dist += SQ(root->pos[i] - pos[i]);
    if (dist < close->dist) {close->node = root; close->dist = dist;}


    // pos is closer to left child
    kd_rect slice = new_rect(rect->min, rect->max, dim);
    if (pos[root->dir] < root->pos[root->dir])
    {
        slice->max[root->dir] = root->pos[root->dir];
        close = find_nearest(root->left, pos, dim, slice, close);
        slice->max[root->dir] = rect->max[root->dir];
        slice->min[root->dir] = root->pos[root->dir];
        close = find_nearest(root->right, pos, dim, slice, close);
    }
    // pos is closer to right child
    else
    {
        slice->min[root->dir] = root->pos[root->dir];
        close = find_nearest(root->right, pos, dim, slice, close);
        slice->min[root->dir] = rect->min[root->dir];
        slice->max[root->dir] = root->pos[root->dir];
        close = find_nearest(root->left, pos, dim, slice, close);
    }
    free_rect(slice);
    return close;
}

kd_query kd_find_nearest(kd_tree tree, float *pos)
{
    kd_query close = new_query(tree->root, FLT_MAX);
    return find_nearest(tree->root, pos, tree->dim, tree->rect, close);
}

static query_stack new_stack_node(kd_query query)
{
    query_stack node = malloc(sizeof(struct query_stack_s));
    node->query = query; node->next = NULL;
    return node;
}

void stack_push(query_stack *stack, kd_query query)
{
    if (!(*stack)) *stack = new_stack_node(query);
    else
    {
        query_stack node = new_stack_node(query);
        // Query distance is smaller than the first element
        if (query->dist < (*stack)->query->dist)
        {
            node->next = *stack; *stack = node;
        }
        else
        {
            // Find where to insert the node
            query_stack temp = *stack;
            while(temp->next && (temp->next->query->dist < query->dist)) temp = temp->next;
            
            // Insert the node
            node->next = temp->next;
            temp->next = node;
        }
    }
}

static void stack_push_tree(query_stack *stack, kd_node root, float *pos, int dim)
{
    if (root->left) stack_push_tree(stack, root->left, pos, dim);
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) dist += SQ(root->pos[i] - pos[i]);
    stack_push(stack, new_query(root, dist));
    if (root->right) stack_push_tree(stack, root->right, pos, dim);
}

kd_query stack_pop(query_stack *stack)
{
    // Do nothing if the stack is empty
    if (!(*stack)) return NULL;

    kd_query query = (*stack)->query;
    query_stack tmp = *stack;
    *stack = (*stack)->next;
    DEALLOC(tmp);
    return query;
}

void free_stack(query_stack stack)
{
    for (query_stack node = stack; node; node = node->next) DEALLOC(node->query);
    DEALLOC(stack);
}

query_stack find_range(kd_node root, float *pos, float range, int dim, kd_rect rect, query_stack stack)
{
    // Fell out of tree
    if (!root) return stack;
    // The cell doesn't overlap query
    if (distance_to_rect(pos, rect, dim) > range) return stack;

    // The query contains the cell
    float dist_min = 0.0f, dist_max = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        dist_min += SQ(pos[i] - rect->min[i]);
        dist_max += SQ(pos[i] - rect->max[i]);
    }
    if (dist_min < range && dist_max < range)
    {
        stack_push_tree(&stack, root, pos, dim); return stack;
    }

    float dist = 0.0f;
    for (int i = 0; i < dim; i++) dist += SQ(root->pos[i] - pos[i]);
    if (dist < range) stack_push(&stack, new_query(root, dist));

    kd_rect slice = new_rect(rect->min, rect->max, dim);
    slice->max[root->dir] = root->pos[root->dir];
    stack = find_range(root->left, pos, range, dim, slice, stack);

    slice->max[root->dir] = rect->max[root->dir];
    slice->min[root->dir] = root->pos[root->dir];
    stack = find_range(root->right, pos, range, dim, slice, stack);
    free_rect(slice);

    return stack;
}

query_stack kd_find_range(kd_tree tree, float *pos, float range)
{
    return find_range(tree->root, pos, SQ(range), tree->dim, tree->rect, NULL);
}
