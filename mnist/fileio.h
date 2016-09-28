//
//  fileio.h
//  mnist
//
//  Created by TU on 2016/09/28.
//  Copyright © 2016年 TU. All rights reserved.
//

#ifndef fileio_h
#define fileio_h

#include <stdio.h>
#include "fileio.h"
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

void FlipLong(unsigned char * ptr);
void read_images(char *filename,int row, int col, unsigned char images[row][col]);
void read_labels(char *filename, int length, unsigned char array[length]);
void print_images(int size, unsigned char images[][size]);
double print_matrix(int row, int col, double mat[row][col], int showrow, int showcol);

#endif /* fileio_h */
