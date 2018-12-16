#ifndef _SEARCH_H
#define _SEARCH_H

#define SEARCH_VERSION 100

typedef unsigned long long UInt64;
typedef unsigned char UInt8;
typedef float DataType;


#ifdef __cplusplus
extern "C" {
#endif


struct search_result
{
    UInt64 index;
    double value;
};

struct search_model
{
    UInt64 dim;        /* feature dimensionality */ 
    UInt64 nimages;    /* number of images in the model */
    DataType *feature_ptr;    /* feature vectors */
};

search_model *load_model(const char *model_file_name, const UInt64 dim, const UInt64 nimages);
void free_model(search_model **model_ptr_ptr);

UInt64 get_dim(const search_model* model_ptr);
UInt64 get_nr_images(const search_model* model_ptr);

void search_knn(const search_model *model, const DataType* query_ptr, const UInt64 k, const int dfunc, search_result *results);

void compute_l2_distance(const struct search_model *model, const DataType* query_ptr, double* dist_values);
void compute_l1_distance(const struct search_model *model, const DataType* query_ptr, double* dist_values);
void compute_chi2_distance(const struct search_model *model, const DataType* query_ptr, double* dist_values);

void print_model(const search_model* model_ptr);
void free_model_contents(search_model* model_ptr);

#ifdef __cplusplus
}
#endif

#endif /* _search_H */


