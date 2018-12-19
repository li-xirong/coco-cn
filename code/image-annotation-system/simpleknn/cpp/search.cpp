#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
using namespace std;

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "search.h"

#ifndef __S_IFMT
#define __S_IFMT 0170000
#endif

#ifndef __S_IFDIR
#define __S_IFDIR 0040000
#endif

inline bool operator<(const search_result &a, const search_result &b)
{
    return a.value < b.value;
}

/*
search_model *load_model(const char *model_file_name, const UInt64 dim, const UInt64 nimages)
{
    using namespace boost::interprocess;

    FILE *fp = fopen(model_file_name, "rb");
    if (0 == fp) {
        fprintf(stderr, "[search.load_model] failed to open model_file %s\n", model_file_name);
        return 0;
    }

    struct stat info;
    stat(model_file_name, &info);
    if( (info.st_mode & __S_IFMT ) == __S_IFDIR) {
        fprintf(stderr, "[search.load_model] %s is a directory\n",  model_file_name);
        fclose(fp);
        return 0;
    }

    //printf("%d %d %d %d\n", sizeof(dim), sizeof(nimages), dim, nimages);

    UInt64 count = dim * nimages;
    search_model *model = new search_model;

    //printf("%d %d\n", sizeof(count), count);

    //fprintf(stdout, "[search.load model] requesting %llu bytes memory ...\n", count * sizeof(DataType));
    model->feature_ptr = new DataType[count];

    if (0 == model->feature_ptr)
    {
        fprintf(stderr, "[search.load_model] Memory error!\n");
        fclose(fp);
        free_model(&model);
        return 0;
    }

    fread((char *)(model->feature_ptr), sizeof(DataType), count, fp);
    fclose(fp);
    model->dim = dim;
    model->nimages = nimages;

    //print_model(model);

    return model;
}
*/

search_model *load_model(const char *model_file_name, const UInt64 dim, const UInt64 nimages)
{
    using namespace boost::interprocess;

    FILE *fp = fopen(model_file_name, "rb");
    if (0 == fp) {
        fprintf(stderr, "[search.load_model] failed to open model_file %s\n", model_file_name);
        return 0;
    }
    fclose(fp);

    struct stat info;
    stat(model_file_name, &info);
    if( (info.st_mode & __S_IFMT ) == __S_IFDIR) {
        fprintf(stderr, "[search.load_model] %s is a directory\n",  model_file_name);
        fclose(fp);
        return 0;
    }

    file_mapping *m_file = new file_mapping(model_file_name, read_only);
    mapped_region *region = new mapped_region(*m_file, read_only);

    UInt64 count = dim * nimages * sizeof(DataType);

    search_model *model = new search_model;
    model->m_file = m_file;
    model->region = region;
    //Get the address of the mapped region
    model->feature_ptr = (DataType*)region->get_address();
    UInt64 region_size  = region->get_size();

    if (0 == model->feature_ptr)
    {
        fprintf(stderr, "[search.load_model] Memory error!\n");
        free_model(&model);
        return 0;
    }

/*    if (count != region_size) {
        fprintf(stderr, "[search.load_model] File size mis-match number of images!\n");
        printf("nimages: %llu\n", nimages);
        printf("dim: %llu\n", dim);
        printf("region_size: %llu\n", region_size);
        printf("count: %llu\n", count);
        free_model(&model);
        return 0;
    }
*/
    model->dim = dim;
    model->nimages = nimages;

    return model;
}

void free_model_contents(search_model* model_ptr)
{
     /*
     if (0 != model_ptr->feature_ptr)
     {
         //fprintf(stdout, "[search.free_model_contents]\n");
         delete [] model_ptr->feature_ptr;
         model_ptr->feature_ptr = 0;
     }
     */
     if (0 != model_ptr->feature_ptr)
     {
         //fprintf(stdout, "[search.free_model_contents]\n");
         delete model_ptr->region;
         delete model_ptr->m_file;
         model_ptr->feature_ptr = 0;
     }
}


void free_model(search_model** model_ptr_ptr)
{
     search_model* model_ptr = *model_ptr_ptr;

     if(0 != model_ptr)
     {
         free_model_contents(model_ptr);
         delete model_ptr;
         model_ptr = 0;
         //fprintf(stdout, "[search.free_model]\n");
     }
}

void print_model(const search_model* model_ptr)
{
    fprintf(stdout, "[search.print_model] %llu images, %llu dims\n", model_ptr->nimages, model_ptr->dim);
}


UInt64 get_dim(const search_model* model_ptr)
{
    return model_ptr->dim;
}

UInt64 get_nr_images(const search_model* model_ptr)
{
    return model_ptr->nimages;
}

// 1 - ((xi * yi) / (norm(x) * norm(y)))

void compute_cosine_distance(const search_model *model, const DataType* query_ptr, double* dist_values)
{
    const DataType *ptr = model->feature_ptr;

    for (UInt64 i=0; i<model->nimages; i++)
    {
        double norm_query = 0;
        double norm_ptr = 0;
        double dist = 0;
        for (UInt64 j=0; j<model->dim; j++)
        {
            norm_query += query_ptr[j] * query_ptr[j];
            norm_ptr += ptr[j] * ptr[j];
            dist += query_ptr[j] * ptr[j];
        }
        ptr += model->dim;
        //fprintf(stdout, "%d %f %f\n", i, dist, sqrt(dist));
        dist_values[i] = 1. - (dist / (sqrt(norm_query) * sqrt(norm_ptr)));
    }
}

void compute_l2_distance(const search_model *model, const DataType* query_ptr, double* dist_values)
{
    const DataType *ptr = model->feature_ptr;

    for (UInt64 i=0; i<model->nimages; i++)
    {
        double dist = 0;
        for (UInt64 j=0; j<model->dim; j++)
        {
            double d = query_ptr[j] - ptr[j];
            dist += (d * d);
            //if (0 == i) fprintf(stdout, "%d %f %f %f\n", j, query_ptr[j], ptr[j], d);
        }
        ptr += model->dim;
        //fprintf(stdout, "%d %f %f\n", i, dist, sqrt(dist));
        dist_values[i] = sqrt(dist);
    }
}

void compute_l1_distance(const search_model *model, const DataType* query_ptr, double* dist_values)
{
    const DataType *ptr = model->feature_ptr;

    for (UInt64 i=0; i<model->nimages; i++)
    {
        double dist = 0;
        for (UInt64 j=0; j<model->dim; j++)
        {
            double d = query_ptr[j] - ptr[j];
            dist += fabs(d);
        }
        ptr += model->dim;
        //fprintf(stdout, "%d %f %f\n", i, dist, sqrt(dist));
        dist_values[i] = dist;
    }
}

/*
 * chi2(x,y) = sum( (xi-yi)^2 / (xi+yi) ) / 2
 */
void compute_chi2_distance(const search_model *model, const DataType* query_ptr, double* dist_values)
{
    const DataType *ptr = model->feature_ptr;
    double dist = 0;
    double d = 0;
    double s = 0;

    for (UInt64 i=0; i<model->nimages; i++)
    {
        dist = 0.0;
        for (UInt64 j=0; j<model->dim; j++)
        {
            d = query_ptr[j] - ptr[j];
            s = query_ptr[j] + ptr[j];
            if (s > 1e-8) {
                dist += (d*d)/s;
            }
        }
        dist /= 2;
        ptr += model->dim;
        //fprintf(stdout, "%d %f %f\n", i, dist, sqrt(dist));
        dist_values[i] = dist;
    }
}


void search_knn(const struct search_model *model, const DataType* query_ptr, const UInt64 k, const int dfunc, struct search_result *results)
{
    double * dist_values = new double[model->nimages];

    switch (dfunc) {
        case 0:
            compute_l1_distance(model, query_ptr, dist_values);
            break;
        case 2:
            compute_chi2_distance(model, query_ptr, dist_values);
            break;
        case 4:
            compute_cosine_distance(model, query_ptr, dist_values);
            break;
        default:
            compute_l2_distance(model, query_ptr, dist_values);
            break;
    }
    /*if (1 == l2) {
        compute_distance(model, query_ptr, dist_values);
    } else {
        compute_l1_distance(model, query_ptr, dist_values);
    } */
    struct search_result *tosort = new search_result[model->nimages];
    for (UInt64 i=0; i<model->nimages; i++)
    {
        tosort[i].index = i;
        tosort[i].value = dist_values[i];
    }
    delete [] dist_values;

    if (k <= (model->nimages >> 1)) {
        partial_sort(tosort, tosort + k, tosort + model->nimages);
    }
    else {
        sort(tosort, tosort + model->nimages);
    }

    for (UInt64 i=0; i<k; i++)
    {
        results[i].index = tosort[i].index;
        results[i].value = tosort[i].value;
    }
    delete [] tosort;
}
