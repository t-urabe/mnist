//
//  filecast.h
//  mnist
//
//  Created by TU on 2016/09/28.
//  Copyright © 2016年 TU. All rights reserved.
//

#ifndef filecast_h
#define filecast_h

#include <stdio.h>
#include <stdlib.h>

void matUCtoD(int rowx, int colx, unsigned char x[rowx][colx],
              int rowx_d, int colx_d, double x_d[rowx_d][colx_d]);
void label_oh(int len, unsigned char y[len],
              int rowy_oh, int coly_oh, double y_oh[rowy_oh][coly_oh]);
#endif /* filecast_h */
