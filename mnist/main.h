//
//  main.h
//  mnist
//
//  Created by TU on 2016/09/28.
//  Copyright © 2016年 TU. All rights reserved.
//

#ifndef main_h
#define main_h

#include <stdio.h>
#include <stdlib.h> // exit()
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include "fileio.h"
#include "filecast.h"
#include "network.h"
int minibatch(int startPoint, int num_minibatch, int num_images, int class, int col,
               double x[num_images][col], double x_mini[num_minibatch][col],
               double y[num_images][class], double y_mini[num_images][class]);
double datain2refine(int num_images, int size, int num_hidden, int class, double x[num_images][size],
                   double w[size][num_hidden], double b[num_hidden][1], double h[num_images][num_hidden],
                   double z[num_images][num_hidden], double w_s[num_hidden][class], double b_s[class][1],
                   double y_[num_images][class], double d[num_images][class],
                   double delta0[num_images][num_hidden], double delta1[num_images][class]);
 

#endif /* main_h */
