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

#include <stdio.h>
#include <stdlib.h> // exit()
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#define TRAIN_X_PATH "data/train-images-idx3-ubyte"
#define TRAIN_Y_PATH "data/train-labels-idx1-ubyte"
#define TEST_X_PATH "data/t10k-images-idx3-ubyte"
#define TEST_Y_PATH "data/t10k-labels-idx1-ubyte"

#define NUM_IMAGES_TRAIN 60000
#define NUM_IMAGES_TEST 10000
#define SIZE 784   /* 28 x 28 */
#define NUM_HIDDEN 5
#define CLASS 10

static int num[10];
unsigned char train_x[NUM_IMAGES_TRAIN][SIZE];
double train_x_d[NUM_IMAGES_TRAIN][SIZE];
unsigned char test_x[NUM_IMAGES_TEST][SIZE];
double test_x_d[NUM_IMAGES_TEST][SIZE];
unsigned char train_y[NUM_IMAGES_TRAIN];
double train_y_oh[NUM_IMAGES_TRAIN][CLASS]; // one-hot label
unsigned char test_y[NUM_IMAGES_TEST];
double test_y_oh[NUM_IMAGES_TEST][CLASS];      // one-hot label

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

void FlipLong(unsigned char * ptr);
void read_images(char *filename,int row, int col, unsigned char images[row][col]);
void read_labels(char *filename, int length, unsigned char array[length]);
void print_images(unsigned char images[][SIZE]);
void matUCtoD(int rowx, int colx, unsigned char x[rowx][colx],
              int rowx_d, int colx_d, double x_d[rowx_d][colx_d]);
void label_oh(int len, unsigned char y[len],
              int rowy_oh, int coly_oh, double y_oh[rowy_oh][coly_oh]);
void matmul(int row1, int col1, double fMat1[row1][col1],
            int row2, int col2, double fMat2[row2][col2],
            int row3, int col3, double tMat3[row3][col3]);
void linCon_tanh(int rowx, int colx, double x[rowx][colx],
                 int roww, int colw, double w[roww][colw],
                 int rowb, int colb, double b[rowb][colb],
                 int rowz, int colz, double z[rowz][colz]);
void linCon_softmax(int rowx, int colx, double x[rowx][colx],
                    int roww, int colw, double w[roww][colw],
                    int rowb, int colb, double b[rowb][colb],
                    int rowz, int colz, double z[rowz][colz]);
void backprop_tanh(int rowu, int colu, double u[rowu][colu],
                   int roww, int colw, double w[roww][colw],
                   int rowd1, int cold1, double d1[rowd1][cold1],
                   int rowd0, int cold0, double d0[rowd0][cold0]);
void init(unsigned int range,
          int roww, int colw, double w[roww][colw],
          int rowb, int colb, double b[rowb][colb]);
 
    
    
/*************************
    main function
 ************************/
    
int main(int argc, char **argv){
    
    /* read images and labels for train and test data */
    read_images(TRAIN_X_PATH, NUM_IMAGES_TRAIN, SIZE, train_x);
    read_images(TEST_X_PATH, NUM_IMAGES_TEST, SIZE,  test_x);
    read_labels(TRAIN_Y_PATH, NUM_IMAGES_TRAIN, train_y);
    read_labels(TEST_Y_PATH, NUM_IMAGES_TEST, test_y);
    
    /* cast image data from unsigned char to double */
    matUCtoD(NUM_IMAGES_TRAIN, SIZE, train_x, NUM_IMAGES_TRAIN,SIZE, train_x_d);
    matUCtoD(NUM_IMAGES_TEST, SIZE, test_x, NUM_IMAGES_TRAIN,SIZE, test_x_d);
    
    /* make one-hot label matrix */
    double train_y_oh[NUM_IMAGES_TRAIN][CLASS];
    double test_y_oh[NUM_IMAGES_TEST][CLASS];
    label_oh(NUM_IMAGES_TRAIN, train_y, NUM_IMAGES_TRAIN, CLASS, train_y_oh);
    label_oh(NUM_IMAGES_TEST, test_y, NUM_IMAGES_TEST, CLASS, test_y_oh);
    
    /* initialize variables in hidden layer */
    double w[SIZE][NUM_HIDDEN];
    double b[NUM_HIDDEN][1];
    init(5, SIZE, NUM_HIDDEN ,w, NUM_HIDDEN,1,b);
    
    /* initialize variables in surface layer */
    double w_s[NUM_HIDDEN][CLASS];
    double b_s[CLASS][1];
    init(5, NUM_HIDDEN, CLASS, w_s, CLASS, 1, b_s);
    
    /* allocate z (output in hidden layer) */
    double (*z)[NUM_HIDDEN];
    z = calloc(NUM_IMAGES_TRAIN*NUM_HIDDEN, sizeof(double));
    
    /* forward propagation through hidden layer */
    linCon_tanh(NUM_IMAGES_TRAIN, SIZE, train_x_d, SIZE, NUM_HIDDEN, w, NUM_HIDDEN, 1, b, NUM_IMAGES_TRAIN, NUM_HIDDEN, z);
    
    /* allocate y_ (output in surface layer) */
    double (*y_)[CLASS];
    y_ = calloc(NUM_IMAGES_TRAIN*CLASS, sizeof(double));
    
    /* forward propagation through surface layer */
    linCon_softmax(NUM_IMAGES_TRAIN, NUM_HIDDEN, z, NUM_HIDDEN, CLASS, w_s, CLASS, 1, b_s, NUM_IMAGES_TRAIN, CLASS, y_);
    
    //double cross = cross_entropy(NUM_IMAGES_TRAIN, CLASS, y_, NUM_IMAGES_TRAIN, train_y);
    
    //printf("cross: %e\n", (cross));
    
    
    
    return 1;
}
   
   
    
/* http://www.kk.iij4u.or.jp/~kondo/wave/swab.html */
void FlipLong(unsigned char * ptr) {
    register unsigned char val;
    
    /* Swap 1st and 4th bytes */
    val = *(ptr);
    *(ptr) = *(ptr+3);
    *(ptr+3) = val;
    
    /* Swap 2nd and 3rd bytes */
    ptr += 1;
    val = *(ptr);
    *(ptr) = *(ptr+1);
    *(ptr+1) = val;
}


/*  Read the training images into memory */
void read_images(char *filename,int row, int col, unsigned char images[row][col]){
    
    int i, fd;
    unsigned char *ptr;
    
    if ((fd=open(filename, O_RDONLY))==-1){
        printf("couldn't open image file: %s\n", filename);
        exit(0);
    }
    
    read(fd,num, 4*sizeof(int) );
    
    for (i=0; i<4; i++) {
        ptr  = (unsigned char *)(num + i);
        FlipLong( ptr);
        printf("%d\n", num[i]);
        ptr = ptr + sizeof(int);
    }
    
    for (i=0; i<row; i++) {
        read(fd, images[i], col*sizeof(unsigned char) );
    }
    close(fd);
}


/*  Read the training images into memory */
void read_labels(char *filename, int length, unsigned char array[length]){
    
    int i, fd;
    unsigned char *ptr;
    
    if ((fd=open(filename,O_RDONLY))==-1){
        printf("couldn't open image file");
        exit(0);
    }
    
    read(fd,num, 2*sizeof(int) );
    
    for (i=0; i<2; i++) {
        ptr  = (unsigned char *)(num + i);
        FlipLong( ptr);
        printf("%d\n", num[i]);
        ptr = ptr + sizeof(int);
    }
    
    for (i=0; i<length; i++) {
        read(fd,  &array[i], sizeof(unsigned char) );
    }
    close(fd);
}


void print_images(unsigned char images[][SIZE]){
    
    int i, j;
    
    for (i=0; i<10; i++) {
        printf("#####   image %d/%d\n",i, NUM_IMAGES_TEST);
        for (j=0; j<SIZE; j++) {
            printf("%3d ", images[i][j]);
            if ( (j+1) % 28 == 0 ){
                printf("\n");
            }
        }
        printf("\n\n");
    }
    
}

void matUCtoD(int rowx, int colx, unsigned char x[rowx][colx],
              int rowx_d, int colx_d, double x_d[rowx_d][colx_d]){
    int i, j;
    for (i=0; i<rowx; i++){
        for(j=0; j<colx;j++)
            x_d[i][j] = (double)(x[i][j]);
    }
}

void label_oh(int len, unsigned char y[len],
              int rowy_oh, int coly_oh, double y_oh[rowy_oh][coly_oh]){
    if(len != rowy_oh){
        printf("length did not match in label_oh func");
        exit(0);
    }
    for (int i=0; i<len; i++)
        y_oh[i][y[i]] = 1.0;
}

void matmul(int row1, int col1, double fMat1[row1][col1],
            int row2, int col2, double fMat2[row2][col2],
            int row3, int col3, double tMat3[row3][col3]){
    if (col1 != row2 || row1 != row3 || col2 != col3){
        printf("Mat dimension did not match\n");
        exit(0);
    }
    
    for (int i=0; i<row3; i++){
        for(int j=0; j<col3; j++){
            for (int k=0; k<col1; k++)
                tMat3[i][j] += fMat1[i][k] * fMat2[k][j];
        }
    }
}

void linCon_tanh(int rowx, int colx, double x[rowx][colx],
                 int roww, int colw, double w[roww][colw],
                 int rowb, int colb, double b[rowb][colb],
                 int rowz, int colz, double z[rowz][colz]){
    if (colx != roww || rowx != rowz || colw != colz || colw != rowb || colb != 1){
        printf("Mat shape did not match\n");
        exit(0);
    }
    int i,j,k;
    double (*h)[colw];
    h = calloc(rowx*colw, sizeof(double));
    for (i =0; i<rowz; i++){
        for(j=0; j<colz; j++){
            for(k=0; k<colx; k++){
                h[i][j] += x[i][k] * w[k][j];
                //printf("i =%d, j= %d, k= %d\n", i,j,k);
                //printf("x[%d][%d] = %f\n", i, k, (double)(x[i][k]));
            }
            //printf("h[%d][%d]= %e\n", i, j, h[i][j]);
            h[i][j] += b[j][0];
            z[i][j] = tanh(h[i][j]);
            //printf("z[%d][%d]= %e\n", i, j, z[i][j]);
        }
    }
    free(h);
}

void linCon_softmax(int rowx, int colx, double x[rowx][colx],
                    int roww, int colw, double w[roww][colw],
                    int rowb, int colb, double b[rowb][colb],
                    int rowz, int colz, double z[rowz][colz]){
    if (colx != roww || rowx != rowz || colw != colz || colw != rowb || colb != 1){
        printf("Mat shape did not match\n");
        exit(0);
    }
    int i,j,k;
    
    double (*h)[colw];
    h = calloc(rowx*colw, sizeof(double));
    for (i =0; i<rowz; i++){
        for(j=0; j<colz; j++){
            for(k=0; k<colx; k++){
                h[i][j] += x[i][k] * w[k][j];
            }
            //printf("h[%d][%d]= %e\n", i, j, h[i][j]);
            h[i][j] += b[j][0];
        }
    }
    
    for (int i=0; i<rowz; i++){
        double prob[colz];
        double sum = 0;
        
        for(int j=0; j<colz; j++){
            //printf("atest: %f\n", exp(h[i][j]));
            prob[j] = exp(h[i][j]);
            sum += prob[j];
        }
        for(int j=0; j<colz; j++){
            z[i][j] = prob[j]/sum;
        }
    }
    free(h);
}

void backprop_tanh(int rowu, int colu, double u[rowu][colu],
                   int roww, int colw, double w[roww][colw],
                   int rowd1, int cold1, double d1[rowd1][cold1],
                   int rowd0, int cold0, double d0[rowd0][cold0]){
    if (cold1 != roww || colu != colw || cold0 != colw){
        printf ("Mat shape did not match\n");
        exit(0);
    }
    
    double (*temp)[colw];
    temp = calloc(rowd1*colw, sizeof(double));
    for (int i=0; i< rowd1; i++ ){
        for(int j=0; j<colw; j++){
            for(int k=0; k<cold1; k++){
                temp[i][j] += d1[i][k] * w[k][j];
            }
        }
    }
    
    for(int i=0; i< rowd0; i++){
        for(int j=0; j<cold0; j++){
            d0[i][j] = (1- pow(tanh(u[i][j]), 2.0)) * temp[i][j];
        }
    }
}

void init(unsigned int range,
          int roww, int colw, double w[roww][colw],
          int rowb, int colb, double b[rowb][colb]){
    if (colw != rowb){
        printf("Matrix shape did not match");
        exit(0);
    }
    
    for (int i=0; i<roww; i++){
        for(int j=0; j<colw; j++){
            w[i][j] = ((double)rand()/RAND_MAX - 0.5)*range/2.0;
        }
    }
    
    for (int j =0; j<colw; j++){
        b[j][0] = 0;
    }
}

/*
double cross_entropy(int rowy_, int coly_, double y_[rowy_][coly_],
                     int lend, unsigned char d[lend],
                     int lend, unsigned char d[lend]){
    //深層学習　p51見て引数を再考
    double e;
    for(int i=0; i<rowy_; i++){
        for(int j=0; j<coly_; j++){
            if (d[i] == j)
                e -= log(y_[i][j]);
        }
    }
    return e;
}
*/
