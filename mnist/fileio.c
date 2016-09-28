//
//  fileio.c
//  mnist
//
//  Created by TU on 2016/09/28.
//  Copyright © 2016年 TU. All rights reserved.
//

#include "fileio.h"
#include "main.h"

static int num[10];

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


void print_images(int size, unsigned char images[][size]){
    
    int i, j;
    
    for (i=0; i<10; i++) {
        for (j=0; j<size; j++) {
            printf("%3d ", images[i][j]);
            if ( (j+1) % 28 == 0 ){
                printf("\n");
            }
        }
        printf("\n\n");
    }
    
}

double print_matrix(int row, int col, double mat[row][col], int showrow, int showcol){
    double sum = 0;
    for (int i=0; i<showrow;i++){
        for(int j=0; j<showcol;j++){
            sum  += mat[i][j];
            printf("[%d][%d]= %.3f  ", i,j,mat[i][j]);
        }
        puts("");
    }
    puts("");
    return sum;
}

