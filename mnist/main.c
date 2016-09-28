//
//  main.c
//  mnist
//
//  Created by TU on 2016/09/24.
//  Copyright © 2016年 TU. All rights reserved.
//
/*
 *  MNIST Data file from http://yann.lecun.com/exdb/mnist/
 *  2006.10.25 A.Date
 */

#include "main.h"

#define TRAIN_X_PATH "data/train-images-idx3-ubyte"
#define TRAIN_Y_PATH "data/train-labels-idx1-ubyte"
#define TEST_X_PATH "data/t10k-images-idx3-ubyte"
#define TEST_Y_PATH "data/t10k-labels-idx1-ubyte"
#define NUM_IMAGES_TRAIN 60000
#define NUM_IMAGES_TEST 10000
#define SIZE 784   /* 28 x 28 */
#define NUM_HIDDEN 5
#define CLASS 10
double alpha = 1e-4;

/*
 TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000803(2051) magic number
 0004     32 bit integer  60000            number of images
 0008     32 bit integer  28               number of rows
 0012     32 bit integer  28               number of columns
 0016     unsigned byte   ??               pixel
 0017     unsigned byte   ??               pixel
 ........
 ====     unsigned byte   ??               pixel
 
 Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
 */

/* function prototype decleration */



   
/*************************
    main function
 ************************/
    
int main(int argc, char **argv){
    unsigned char (*train_x)[SIZE];
    train_x = calloc(NUM_IMAGES_TRAIN*SIZE, sizeof(unsigned char));
    double (*train_x_d)[SIZE];
    train_x_d = calloc(NUM_IMAGES_TRAIN*SIZE, sizeof(double));
    unsigned char (*test_x)[SIZE];
    test_x = calloc(NUM_IMAGES_TEST*SIZE, sizeof(unsigned char));
                    
    double (*test_x_d)[SIZE];
    test_x_d = calloc(NUM_IMAGES_TEST*SIZE, sizeof(double));
    
    unsigned char (*train_y);
    train_y = calloc(NUM_IMAGES_TRAIN, sizeof(unsigned char));
    //double train_y_oh[NUM_IMAGES_TRAIN][CLASS]; // one-hot label
    unsigned char (*test_y);
    test_y = calloc(NUM_IMAGES_TEST, sizeof(unsigned char));
    //double test_y_oh[NUM_IMAGES_TEST][CLASS];      // one-hot label

   
    /* read images and labels for train and test data */
    read_images(TRAIN_X_PATH, NUM_IMAGES_TRAIN, SIZE, train_x);
    read_images(TEST_X_PATH, NUM_IMAGES_TEST, SIZE,  test_x);
    read_labels(TRAIN_Y_PATH, NUM_IMAGES_TRAIN, train_y);
    read_labels(TEST_Y_PATH, NUM_IMAGES_TEST, test_y);
    
    /* cast image data from unsigned char to double */
    matUCtoD(NUM_IMAGES_TRAIN, SIZE, train_x, NUM_IMAGES_TRAIN,SIZE, train_x_d);
    matUCtoD(NUM_IMAGES_TEST, SIZE, test_x, NUM_IMAGES_TRAIN,SIZE, test_x_d);
    
    /* make one-hot label matrix */
    double train_y_oh[NUM_IMAGES_TRAIN][CLASS] = {0};
    double test_y_oh[NUM_IMAGES_TEST][CLASS]=  {0};
    label_oh(NUM_IMAGES_TRAIN, train_y, NUM_IMAGES_TRAIN, CLASS, train_y_oh);
    label_oh(NUM_IMAGES_TEST, test_y, NUM_IMAGES_TEST, CLASS, test_y_oh);
   
    /* free memory */
    free(train_x);
    free(test_x);
    free(train_y);
    free(test_y);
    
    
    /* initialize variables in hidden layer */
    double w[SIZE][NUM_HIDDEN];
    double b[NUM_HIDDEN][1];
    init(5, SIZE, NUM_HIDDEN ,w, NUM_HIDDEN,1,b);
    
    /* initialize variables in surface layer */
    double w_s[NUM_HIDDEN][CLASS];
    double b_s[CLASS][1];
    init(1, NUM_HIDDEN, CLASS, w_s, CLASS, 1, b_s);
    
    /* allocate h (intermid value in hidden layer) */
    double (*h)[NUM_HIDDEN];
    h = calloc(NUM_IMAGES_TRAIN*NUM_HIDDEN, sizeof(double));
    /* allocate z (output in hidden layer) */
    double (*z)[NUM_HIDDEN];
    z = calloc(NUM_IMAGES_TRAIN*NUM_HIDDEN, sizeof(double));
    /* allocate y_ (output in surface layer) */
    double (*y_)[CLASS];
    y_ = calloc(NUM_IMAGES_TRAIN*CLASS, sizeof(double));
    double (*delta1)[CLASS];
    delta1 = calloc(NUM_IMAGES_TRAIN*CLASS, sizeof(double));
    
    double (*delta0)[NUM_HIDDEN];
    delta0 = calloc(NUM_IMAGES_TRAIN*NUM_HIDDEN, sizeof(double));
    
    double cross;
    
    
    /******************
        loop
     ******************/
    
    int count = 0;
    while(count <20){
    
        printf("count: %d\n", count);
        
    /* forward propagation through hidden layer */
    linCon(NUM_IMAGES_TRAIN, SIZE, train_x_d, SIZE, NUM_HIDDEN, w, NUM_HIDDEN, 1, b, NUM_IMAGES_TRAIN, NUM_HIDDEN, h);
    tanh_mat(NUM_IMAGES_TRAIN,NUM_HIDDEN, h,
             NUM_IMAGES_TRAIN,NUM_HIDDEN, z);
        
    /* forward propagation through surface layer */
    linCon_softmax(NUM_IMAGES_TRAIN, NUM_HIDDEN, z, NUM_HIDDEN, CLASS, w_s, CLASS, 1, b_s, NUM_IMAGES_TRAIN, CLASS, y_);
        
    /* print answer */
    answer(NUM_IMAGES_TRAIN,CLASS,train_y_oh,NUM_IMAGES_TRAIN,CLASS,y_);
    
    /* calculate cross-entropy */
    cross = cross_entropy(NUM_IMAGES_TRAIN,CLASS, y_,
                                 NUM_IMAGES_TRAIN,CLASS, train_y_oh);
    printf("cross=%.5f\n", cross);
    
    /* calculate delta1 */
    for (int i=0; i< NUM_IMAGES_TRAIN; i++){
        for(int j=0; j< CLASS; j++){
            delta1[i][j] = y_[i][j] - train_y_oh[i][j];
        }
    }
    
    /* back propagation */
    backprop_tanh(NUM_IMAGES_TRAIN, NUM_HIDDEN, z, NUM_HIDDEN, CLASS, w_s, NUM_IMAGES_TRAIN, CLASS, delta1, NUM_IMAGES_TRAIN, NUM_HIDDEN, delta0);
    
    /* sto */
    /* w_sto = (z.T).dot(delta) */
    refine_variables(NUM_HIDDEN, CLASS, w_s, NUM_IMAGES_TRAIN, CLASS, delta1, NUM_IMAGES_TRAIN,NUM_HIDDEN,z, alpha);
    refine_variables(SIZE, NUM_HIDDEN, w, NUM_IMAGES_TRAIN, NUM_HIDDEN, delta0, NUM_IMAGES_TRAIN,SIZE,train_x_d, alpha);
    
    count += 1;
        printf("\n");
    
    }
    
    return 1;
}
   
   

