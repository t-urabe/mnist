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
#define NUM_HIDDEN 10
#define CLASS 10
#define NUM_MINIBATCH 1000
#define SHOWNUM 1
double alpha = 1e-5;
double var = 1e+2;
int useminibatch = 1;
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
    
    /* mini batch */
    double (*train_x_d_mini)[SIZE];
    train_x_d_mini = calloc(NUM_MINIBATCH*SIZE, sizeof(double));
    double (*train_y_oh_mini)[CLASS];
    train_y_oh_mini = calloc(NUM_MINIBATCH*CLASS, sizeof(double));
    double (*test_x_d_mini)[SIZE];
    test_x_d_mini = calloc(NUM_MINIBATCH*SIZE, sizeof(double));
    double (*test_y_oh_mini)[CLASS];
    test_y_oh_mini = calloc(NUM_MINIBATCH*CLASS, sizeof(double));
    
    
    /* initialize variables in hidden layer */
    double (*w)[NUM_HIDDEN];
    w = calloc(SIZE*NUM_HIDDEN, sizeof(double));
    double (*b)[1];
    b =calloc(NUM_HIDDEN*1, sizeof(double));
    
    init(var, SIZE, NUM_HIDDEN ,w, NUM_HIDDEN,1,b);
    
    /* initialize variables in surface layer */
    double (*w_s)[CLASS];
    w_s = calloc(NUM_HIDDEN*CLASS, sizeof(double));
    double (*b_s)[1];
    b_s = calloc(CLASS*1, sizeof(double));
    init(var, NUM_HIDDEN, CLASS, w_s, CLASS, 1, b_s);
    
    /************ full sample **************/
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
    /***************************************/
    
    
    /*************mini batch ****************/
    /* allocate h (intermid value in hidden layer) */
    double (*h_mini)[NUM_HIDDEN];
    h_mini = calloc(NUM_MINIBATCH*NUM_HIDDEN, sizeof(double));
    /* allocate z (output in hidden layer) */
    double (*z_mini)[NUM_HIDDEN];
    z_mini = calloc(NUM_MINIBATCH*NUM_HIDDEN, sizeof(double));
    /* allocate y_ (output in surface layer) */
    double (*y__mini)[CLASS];
    y__mini = calloc(NUM_MINIBATCH*CLASS, sizeof(double));
    double (*delta1_mini)[CLASS];
    delta1_mini = calloc(NUM_MINIBATCH*CLASS, sizeof(double));
    double (*delta0_mini)[NUM_HIDDEN];
    delta0_mini = calloc(NUM_MINIBATCH*NUM_HIDDEN, sizeof(double));
     /**************************************/
    
    
    int startPoint = 0000; /*initialize start index for minibatch */
    double cross;
   
    
    /* apply full-batch learning for the first time */
    /*
    cross = datain2refine(NUM_IMAGES_TRAIN, SIZE, NUM_HIDDEN, CLASS, train_x_d,
                  w, b, h,
                  z,w_s, b_s,
                  y_, train_y_oh_mini,
                  delta0, delta1);
    printf("count: first fullbatch, cross:%.4f ", cross);
    // print answer
    answer(NUM_IMAGES_TRAIN,CLASS,train_y_oh,NUM_IMAGES_TRAIN,CLASS,y_);
    printf("\n");
    */
    
    /******************
        loop
     ******************/
    
    int count = 0;
    while(count <20000){
       //printf("startPoint: %d\n", startPoint);
            int startPointOld = startPoint;
        
            /* minibatch */
            startPoint = minibatch(startPoint, NUM_MINIBATCH, NUM_IMAGES_TRAIN, CLASS, SIZE,
                                   train_x_d, train_x_d_mini,
                                   train_y_oh, train_y_oh_mini);
       
        
        if(useminibatch){
            /*
          
           
            printf("mini\n");
            for(int n=0; n<NUM_MINIBATCH; n++){
                for(int i=0; i<784; i++){
                    if(i %28 ==0){
                        printf("\n");
                    }
                    printf("%4d", (int)train_x_d_mini[n][i]);
                }
                printf("\n\n");
            }
            
            printf("full\n");
            for(int n=0; n<NUM_MINIBATCH; n++){
                for(int i=0; i<784; i++){
                    if(i %28 ==0){
                        printf("\n");
                    }
                    printf("%4d", (int)train_x_d[n+startPointOld][i]);
                }
                printf("\n\n");
            }
            
            printf("mini\n");
            for(int n=0; n<NUM_MINIBATCH; n++){
                for(int i=0; i<10; i++){
                    printf("%4d", (int)train_y_oh_mini[n][i]);
                }
                printf("\n");
            }
            printf("\n\n");
            
            printf("full\n");
            for(int n=0; n<NUM_MINIBATCH; n++){
                for(int i=0; i<10; i++){
                    printf("%4d", (int)train_y_oh[n+startPointOld][i]);
                }
                puts("");
            }
            printf("\n\n");
            
            */
            
            //double total = print_matrix(NUM_MINIBATCH, SIZE, train_x_d_mini, NUM_MINIBATCH, SIZE);
            printf("using minibatch= %d, startPoint: %d\n", NUM_MINIBATCH, startPoint);
            for(int i=0; i< NUM_MINIBATCH;i++)
                for(int j=0; j<SIZE; j++)
                    if(train_x_d[i+startPointOld][j] != train_x_d_mini[i][j]){
                        printf("x[%d][%d]:%f  ,", i+startPointOld,j, train_x_d[i][j]);
                        printf("mini[%d][%d]:%f  ,", i,j, train_x_d_mini[i][j]);
                        printf("error!!!!!!\n");
                    }
            
            
            cross = datain2refine(NUM_MINIBATCH, SIZE, NUM_HIDDEN, CLASS, train_x_d_mini,
                      w, b, h_mini,
                      z_mini,w_s, b_s,
                      y__mini, train_y_oh_mini,
                      delta0_mini, delta1_mini);
        } else {
            cross =datain2refine(NUM_IMAGES_TRAIN, SIZE, NUM_HIDDEN, CLASS, train_x_d,
                      w, b, h,
                      z,w_s, b_s,
                      y_, train_y_oh,
                      delta0, delta1);
        }
        
        if (count % SHOWNUM == 0){
            printf("count: %d, cross:%.4f ", count, cross);
            /* print answer */
            //answer(NUM_IMAGES_TRAIN,CLASS,train_y_oh,NUM_IMAGES_TRAIN,CLASS,y__mini);
            printf("\n");
            answer(NUM_MINIBATCH,CLASS,train_y_oh_mini,NUM_MINIBATCH,CLASS,y__mini);
            printf("\n");
        }
        
        printf("loop end \n\n");
        count += 1;
    
    }
    
    return 1;
}

/* slice matrix for given minibatch */
int minibatch(int startPoint, int num_minibatch, int num_images, int class, int col,
               double x[num_images][col], double x_mini[num_minibatch][col],
               double y[num_images][class], double y_mini[num_images][class]){
    int i,j;
    for(i=0; i<num_minibatch; i++){
        for(j=0; j<col; j++){
            x_mini[i][j] = x[startPoint+i][j];
        }
        for(j=0; j<class; j++){
            y_mini[i][j] = y[startPoint+i][j];
        }
    }
    return (startPoint + 2*num_minibatch < num_images)? startPoint + num_minibatch: 0;
}


double datain2refine(int num_images, int size, int num_hidden, int class, double x[num_images][size],
                   double w[size][num_hidden], double b[num_hidden][1], double h[num_images][num_hidden],
                   double z[num_images][num_hidden], double w_s[num_hidden][class], double b_s[class][1],
                   double y_[num_images][class], double d[num_images][class],
                   double delta0[num_images][num_hidden], double delta1[num_images][class]){
     /* forward propagation through hidden layer */
    linCon(num_images, size, x, size, num_hidden, w, num_hidden, 1, b, num_images, num_hidden, h);
    tanh_mat(num_images,num_hidden, h,
             num_images,num_hidden, z);
        
    /* forward propagation through surface layer */
    linCon_softmax(num_images, num_hidden, z, num_hidden, class, w_s, class, 1, b_s, num_images, class, y_);
        
   
    /* calculate cross-entropy */
    double cross = cross_entropy(num_images,class, y_,
                                 num_images,class, d);
    //printf("cross=%.5f\n", cross);
    
    /* calculate delta1 */
    for (int i=0; i< num_images; i++){
        for(int j=0; j< class; j++){
            delta1[i][j] = y_[i][j] - d[i][j];
        }
    }

    /* back propagation */
    backprop_tanh(num_images, num_hidden, z, num_hidden, class, w_s, num_images, class, delta1, num_images, num_hidden, delta0);
    
    /* sto */
    /* w_sto = (z.T).dot(delta) */
    refine_variables(num_hidden, class, w_s, class, b_s, num_images, class, delta1, num_images,num_hidden,z, alpha);
    refine_variables(size, num_hidden, w, num_hidden, b, num_images, num_hidden, delta0, num_images,size,x, alpha);
    
    return cross;
}

