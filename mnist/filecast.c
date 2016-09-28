//
//  filecast.c
//  mnist
//
//  Created by TU on 2016/09/28.
//  Copyright © 2016年 TU. All rights reserved.
//

#include "filecast.h"
#include "main.h"

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